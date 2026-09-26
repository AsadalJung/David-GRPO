# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import json
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.validation_output import (
    ValidationOutputWriter,
    build_validation_record,
    is_primary_process,
    resolve_validation_output_base_dir,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl',use_observation_mask=False):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]
    if use_observation_mask:
        observation_mask = data.batch['loss_mask']

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        if use_observation_mask:
            kld = kld * observation_mask
        else:
            kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)
        

    token_level_rewards = token_level_scores - beta * kld
    if use_observation_mask:
        current_kl = masked_mean(kld, mask=observation_mask, axis=-1)
        current_kl = torch.mean(current_kl, dim=0).item()
    else:
        current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
        current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, use_observation_mask=False):
    # use_observation_mask is used to mask the observation part of the response, only used in GRPO
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        if use_observation_mask:
            observation_mask = data.batch['loss_mask']
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=observation_mask,
                                                                        index=index)

        else:
            advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'reinforce_plus_plus':
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'remax':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # rollout search budget 스케줄링 상태
        self._current_rollout_max_search = None
        # Validation runs on this single Ray driver. Create the filesystem
        # writer lazily so constructing a trainer never writes to disk.
        self._validation_output_writer = None

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == 'gae':
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'reinforce_plus_plus':
            self.use_critic = False
        elif self.config.algorithm.adv_estimator == 'remax':
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()
        # self.observation_start_seq = self.tokenizer.encode('<search result', add_special_tokens=False)
        # self.observation_end_seq = self.tokenizer.encode('search_result>\n\n', add_special_tokens=False)
        
        self.observation_start_str = '<search_result>'
        self.observation_end_str = '</search_result>'
        
    # def find_observation_boundaries(self, response):
    #     """
    #     查找observation的开始和结束标记，可能出现多个observation，也可能没有observation
        
    #     Args:
    #         response: 响应的token ids，形状 [response_length]
        
    #     Returns:
    #         matched_observation_start: observation开始位置的列表
    #         matched_observation_end: observation结束位置的列表
    #     """
    #     response_length = len(response)
    #     observation_start = []
    #     observation_end = []
        

        
    #     # 查找所有可能的开始标记
    #     for i in range(response_length - len(self.observation_start_seq) + 1):
    #         if all(response[i+j] == self.observation_start_seq[j] for j in range(len(self.observation_start_seq))):
    #             observation_start.append(i)
        
    #     # 查找所有可能的结束标记
    #     for i in range(response_length - len(self.observation_end_seq) + 1):
    #         if all(response[i+j] == self.observation_end_seq[j] for j in range(len(self.observation_end_seq))):
    #             observation_end.append(i + len(self.observation_end_seq))
        
    #     print(observation_start, observation_end)
    #     # 匹配开始和结束标记
    #     matched_observation_start = []
    #     matched_observation_end = []
        
    #     # 为每个开始标记匹配第一个合适的结束标记
    #     for start in observation_start:
    #         # 找到第一个在start之后的结束标记
    #         valid_ends = [end for end in observation_end if end > start]
    #         if valid_ends:
    #             matched_observation_start.append(start)
    #             matched_observation_end.append(min(valid_ends))  # 取最近的结束标记
        
    #     # TODO： 可能还是会有问题，因为observation_start_seq和observation_end_seq可能会有重叠错误。
        
    #     return matched_observation_start, matched_observation_end

    def identify_observation_mask(self, responses, response_mask):
        """
        通过字符串匹配识别响应中的observation部分，创建一个掩码，
        其中observation部分为0，其他部分保持原掩码值
        
        Args:
            responses: 响应的token ids，形状 [batch_size, response_length]
            response_mask: 原始响应掩码，形状 [batch_size, response_length]
        
        Returns:
            loss_mask: 修改后的掩码，observation部分为0
        """
        # 创建一个新的掩码，初始值与response_mask相同
        loss_mask = response_mask.clone()
        
        batch_size, seq_len = responses.size()
        
        for i in range(batch_size):
            # 解码当前响应为文本
            response_text = self.tokenizer.decode(responses[i], skip_special_tokens=False)
            # assert self.tokenizer.encode(response_text, add_special_tokens=False) == responses[i]
            # print(response_text[:5000])
            # print(self.observation_start_str in response_text)
            # print(self.observation_end_str in response_text)
            # 查找所有observation的开始和结束位置
            starts = []
            ends = []
            start_idx = 0
            
            while True:
                start_pos = response_text.find(self.observation_start_str, start_idx)
                # print(start_pos)
                if start_pos == -1:
                    break
                    
                end_pos = response_text.find(self.observation_end_str, start_pos)
                # print(end_pos)
                if end_pos == -1:
                    break
                
                    
                # 找到了一对完整的标记
                starts.append(start_pos)
                # 加上结束标记的长度，使结束位置在标记之后
                ends.append(end_pos + len(self.observation_end_str))
                
                # 从end_pos之后继续搜索
                start_idx = end_pos + 1
            # print(starts, ends)
            # input()
            # 如果找到了匹配的标记对
            for start, end in zip(starts, ends):
                # 将文本中的位置映射到token序列中的索引
                # 为了精确匹配，需要先截取相应文本，然后转换为token来确定准确位置
                
                # 获取开始位置之前的所有文本对应的token数量
                prefix_text = response_text[:start]
                prefix_tokens = len(self.tokenizer.encode(prefix_text, add_special_tokens=False))
                
                # 获取整个observation文本对应的token数量
                observation_text = response_text[start:end]
                observation_tokens = len(self.tokenizer.encode(observation_text, add_special_tokens=False))
                
                # 计算token序列中的开始和结束索引
                token_start_idx = prefix_tokens
                token_end_idx = token_start_idx + observation_tokens
                
                # 确保索引在有效范围内
                token_start_idx = min(token_start_idx, seq_len - 1)
                token_end_idx = min(token_end_idx, seq_len)
                
                # 将observation部分的掩码设为0
                if token_start_idx < token_end_idx:
                    loss_mask[i, token_start_idx:token_end_idx] = 0.0
                # 使用这个mask 选取原有的 response_id，decode 之后打印出来，看看是否正确
                # print(self.tokenizer.decode(responses[i, token_start_idx:token_end_idx], skip_special_tokens=False))
        return loss_mask

    # def identify_observation_mask(self, responses, response_mask):
    #     """
    #     识别响应中出现的所有的observation（搜索结果）部分，创建一个新的掩码，
    #     其中observation部分为0，其他部分保持原掩码值
        
    #     Args:
    #         responses: 响应的token ids，形状 [batch_size, response_length]
    #         response_mask: 原始响应掩码，形状 [batch_size, response_length]
        
    #     Returns:
    #         loss_mask: 修改后的掩码，observation部分为0
    #     """
    #     # if response_mask is None:
    #     #     response_mask = torch.ones_like(responses, dtype=torch.float)
            
    #     # 创建一个新的掩码，初始值与response_mask相同
    #     loss_mask = response_mask.clone()
        
    #     batch_size, seq_len = responses.size()
    #     for i in range(batch_size):
    #         # 查找observation开始和结束的标记
    #         starts, ends = self.find_observation_boundaries(responses[i])
    #         print(starts, ends)
    #         # 如果找到了匹配的开始和结束标记，将observation部分的掩码设为0
    #         for start, end in zip(starts, ends):
    #             if 0 <= start < end <= seq_len:
    #                 loss_mask[i, start:end] = 0.0
                    
    #     return loss_mask

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        print("[validate_config] All configuration checks passed successfully!")

    def _set_rollout_max_search(self, target: int):
        """rollout 최대 검색 횟수를 설정하고 캐시를 갱신한다."""
        if not hasattr(self, 'actor_rollout_wg'):
            return
        try:
            target = int(target)
        except Exception:
            return
        if target != self._current_rollout_max_search:
            self._current_rollout_max_search = target
            self.actor_rollout_wg.set_rollout_max_search(target)

    def _maybe_update_rollout_max_search(self):
        """use_progressive_max_search가 활성화된 경우 max_search_nums를 단계적으로 올린다."""
        if not hasattr(self, 'actor_rollout_wg'):
            return

        rollout_cfg = self.config.actor_rollout_ref.rollout
        if not rollout_cfg.get('use_progressive_max_search', False):
            return

        max_cap = int(rollout_cfg.max_search_nums)
        start = int(rollout_cfg.get('progressive_max_search_start', max_cap))
        interval = int(rollout_cfg.get('progressive_max_search_interval', 20))

        start = max(1, min(start, max_cap))
        interval = max(1, interval)

        # global_steps는 1부터 증가 (resume 시에는 로드된 값 사용)
        steps_since_start = max(0, self.global_steps - 1)
        target = min(max_cap, start + steps_since_start // interval)

        self._set_rollout_max_search(target)

    def _create_dataloader(self):
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         prompt_key=self.config.data.prompt_key,
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         add_gold_sequence=self.config.data.get('add_gold_sequence', False),
                                         gold_response_key=self.config.data.get('gold_response_key', 'gold_response'))
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.config.data.train_batch_size,
                                           drop_last=True,
                                           collate_fn=collate_fn,
                                           sampler=sampler)

        persist_validation_outputs = self.config.trainer.get('persist_validation_outputs', True)
        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       prompt_key=self.config.data.prompt_key,
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       # Preserve the original user message for the
                                       # validation answer audit log. This does not
                                       # alter the tokenized model input.
                                       return_raw_chat=(
                                           self.config.data.get('return_raw_chat', False)
                                           or persist_validation_outputs
                                       ),
                                       truncation='error')
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size=len(self.val_dataset),
                                         shuffle=True,
                                         drop_last=True,
                                         collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f'Size of train dataloader: {len(self.train_dataloader)}')
        print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations_to_wandb(self, inputs, outputs, scores):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        import wandb
        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum([[f"input_{i+1}", f"output_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])

        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _get_validation_output_writer(self):
        """Return the run-local writer on the single primary driver."""

        if not self.config.trainer.get('persist_validation_outputs', True):
            return None
        if not is_primary_process():
            return None
        if self._validation_output_writer is not None:
            return self._validation_output_writer

        configured_dir = self.config.trainer.get('validation_outputs_dir', None)
        base_dir = resolve_validation_output_base_dir(
            default_local_dir=self.config.trainer.default_local_dir,
            configured_dir=configured_dir,
        )
        val_files = self.config.data.val_files
        if isinstance(val_files, str):
            val_files = [val_files]
        else:
            try:
                val_files = list(val_files)
            except TypeError:
                val_files = [str(val_files)]

        self._validation_output_writer = ValidationOutputWriter(
            base_dir,
            session_metadata={
                'project_name': str(self.config.trainer.project_name),
                'experiment_name': str(self.config.trainer.experiment_name),
                'validation_files': val_files,
                'default_local_dir': str(self.config.trainer.default_local_dir),
                'writer_process': 'ray_ppo_driver',
            },
        )
        print(
            f"Validation answers will be persisted under "
            f"{self._validation_output_writer.session_dir}"
        )
        return self._validation_output_writer

    @staticmethod
    def _validation_row_metadata(non_tensor_batch, row_index):
        """Extract one gathered row without assuming every value is a numpy array."""

        metadata = {}
        for key, values in non_tensor_batch.items():
            try:
                metadata[key] = values[row_index]
            except (IndexError, KeyError, TypeError):
                # Keep persistence compatible with optional scalar metadata.
                metadata[key] = values
        return metadata

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        validation_records = []
        validation_writer = self._get_validation_output_writer()

        # 검증에서는 10회만 사용하고, 검증 직후에는 기존 설정으로 복구한다.
        prev_rollout_max_search = self._current_rollout_max_search
        if prev_rollout_max_search is None:
            try:
                prev_rollout_max_search = int(self.config.actor_rollout_ref.rollout.max_search_nums)
            except Exception:
                prev_rollout_max_search = None

        # 검증 시에는 검색 횟수를 항상 10으로 고정
        self._set_rollout_max_search(10)

        try:
            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                    return {}

                # Store original inputs
                input_ids = test_batch.batch['input_ids']
                input_texts = [
                    self.tokenizer.decode(
                        ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for ids in input_ids
                ]
                sample_inputs.extend(input_texts)

                test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }

                # pad to be divisible by dp_size
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                print('validation generation end')

                # Store generated outputs
                output_ids = test_output_gen_batch.batch['responses']
                output_texts = [
                    self.tokenizer.decode(
                        ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for ids in output_ids
                ]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)

                # evaluate using reward_function
                reward_tensor = self.val_reward_fn(test_batch)

                # Store scores
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                if validation_writer is not None:
                    if len(input_texts) != len(output_texts) or len(output_texts) != len(scores):
                        raise RuntimeError(
                            "Validation persistence requires one gathered output and score per input: "
                            f"inputs={len(input_texts)}, outputs={len(output_texts)}, scores={len(scores)}"
                        )
                    for batch_row, (input_text, output_text, score) in enumerate(
                        zip(input_texts, output_texts, scores)
                    ):
                        row_metadata = self._validation_row_metadata(
                            test_batch.non_tensor_batch,
                            batch_row,
                        )
                        validation_records.append(
                            build_validation_record(
                                validation_row=len(validation_records),
                                prompt=input_text,
                                response=output_text,
                                total_reward=score,
                                example_metadata=row_metadata,
                            )
                        )

                reward_tensor_lst.append(reward_tensor)
                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

            if validation_writer is not None:
                output_path = validation_writer.write_step(
                    global_step=self.global_steps,
                    records=validation_records,
                )
                print(
                    f"Persisted {len(validation_records)} complete validation answers "
                    f"to {output_path}"
                )

            self._maybe_log_val_generations_to_wandb(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

            reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
            data_sources = np.concatenate(data_source_lst, axis=0)

            # evaluate test_score based on data source
            data_source_reward = {}
            for i in range(reward_tensor.shape[0]):
                data_source = data_sources[i]
                if data_source not in data_source_reward:
                    data_source_reward[data_source] = []
                data_source_reward[data_source].append(reward_tensor[i].item())

            metric_dict = {}
            for data_source, rewards in data_source_reward.items():
                metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

            return metric_dict
        finally:
            if prev_rollout_max_search is not None:
                self._set_rollout_max_search(prev_rollout_max_search)

    def _replace_with_gold_sequence(self, batch):
        """점수 기반 골드 시퀀스 대체 로직"""
        if not self.config.data.get('add_gold_sequence', False):
            return batch

        gold_tokens = batch.batch.get('gold_response_tokens', None)
        has_gold = batch.non_tensor_batch.get('has_gold_sequence', None)
        
        if gold_tokens is None or has_gold is None:
            return batch

        responses = batch.batch['responses']
        input_ids = batch.batch['input_ids']
        position_ids = batch.batch['position_ids']
        attention_mask = batch.batch['attention_mask']
        prompts = batch.batch['prompts']
        
        # 골드 시퀀스 표시를 위한 플래그 배열 초기화
        batch_size = responses.shape[0]
        is_gold_sequence_flags = [False] * batch_size
        batch.non_tensor_batch['is_gold_sequence_flags'] = np.array(is_gold_sequence_flags, dtype=object)
        
        # 그룹 정보 수집
        use_otr_sampling = self.config.actor_rollout_ref.rollout.get('use_otr_sampling', False)
        
        if use_otr_sampling and 'uid' in batch.non_tensor_batch:
            # UID 기반 그룹 분석
            uid_groups = {}  # {original_uid: [sequence_indices]}
            uids = batch.non_tensor_batch['uid']
            
            for i, uid in enumerate(uids):
                if uid not in uid_groups:
                    uid_groups[uid] = []
                uid_groups[uid].append(i)
            
            groups_info = []  # [(group_idx, sequence_indices)]
            for group_idx, (uid, sequence_indices) in enumerate(uid_groups.items()):
                if group_idx < len(has_gold) and has_gold[group_idx]:
                    groups_info.append((group_idx, sequence_indices))
        else:
            # 기존 고정 그룹 크기 방식
            n = self.config.actor_rollout_ref.rollout.n
            num_groups = responses.shape[0] // n
            
            groups_info = []  # [(group_idx, sequence_indices)]
            for i in range(num_groups):
                group_start = i * n
                group_end = (i + 1) * n
                
                # 그룹 내에서 골드가 있는지 확인
                has_gold_in_group = False
                gold_sample_idx = None
                for j in range(group_start, group_end):
                    if j < len(has_gold) and has_gold[j]:
                        has_gold_in_group = True
                        gold_sample_idx = j
                        break
                
                if has_gold_in_group:
                    sequence_indices = list(range(group_start, group_end))
                    groups_info.append((gold_sample_idx, sequence_indices))
        
        # 각 그룹별로 점수 기반 골드 대체 수행
        gold_replacement_count = 0
        
        for group_idx, sequence_indices in groups_info:
            # 1. 현재 그룹의 생성된 시퀀스들의 점수 계산
            group_responses = responses[sequence_indices]
            group_input_ids = input_ids[sequence_indices]
            group_attention_mask = attention_mask[sequence_indices]
            
            # 임시 배치 생성 (현재 그룹만)
            temp_batch_dict = {}
            for key, tensor in batch.batch.items():
                if isinstance(tensor, torch.Tensor) and tensor.size(0) == batch_size:
                    temp_batch_dict[key] = tensor[sequence_indices]
                else:
                    temp_batch_dict[key] = tensor
            
            from tensordict import TensorDict
            temp_batch = DataProto()
            temp_batch.batch = TensorDict(temp_batch_dict, batch_size=len(sequence_indices))
            temp_batch.non_tensor_batch = {
                key: (values[sequence_indices] if hasattr(values, '__len__') and len(values) == batch_size else values)
                for key, values in batch.non_tensor_batch.items()
            }
            
            # 생성된 시퀀스들 점수 계산
            original_scores = self.reward_fn(temp_batch)
            sequence_scores = original_scores.sum(-1)  # (group_size,)
            
            # 2. 골드 시퀀스 점수 계산
            # 골드로 대체할 임시 시퀀스 생성 (첫 번째 시퀀스를 골드로 임시 대체)
            temp_responses = group_responses.clone()
            temp_input_ids = group_input_ids.clone()
            temp_position_ids = batch.batch['position_ids'][sequence_indices].clone()
            temp_attention_mask = group_attention_mask.clone()
            temp_prompts = prompts[sequence_indices]
            
            # 첫 번째 시퀀스를 골드로 임시 대체
            self._replace_single_sequence(0, group_idx, gold_tokens, temp_responses, temp_input_ids, 
                                        temp_position_ids, temp_attention_mask, temp_prompts)
            
            # 골드 점수 계산용 임시 배치 생성
            gold_batch_dict = {}
            for key, tensor in temp_batch_dict.items():
                if key in ['responses', 'input_ids', 'position_ids', 'attention_mask']:
                    # 이미 위에서 temp_로 수정된 것들 사용
                    if key == 'responses':
                        gold_batch_dict[key] = temp_responses[:1]
                    elif key == 'input_ids':
                        gold_batch_dict[key] = temp_input_ids[:1]
                    elif key == 'position_ids':
                        gold_batch_dict[key] = temp_position_ids[:1]
                    elif key == 'attention_mask':
                        gold_batch_dict[key] = temp_attention_mask[:1]
                else:
                    # 다른 텐서들은 첫 번째 요소만 가져오기
                    if isinstance(tensor, torch.Tensor) and tensor.size(0) > 1:
                        gold_batch_dict[key] = tensor[:1]
                    else:
                        gold_batch_dict[key] = tensor
            
            gold_temp_batch = DataProto()
            gold_temp_batch.batch = TensorDict(gold_batch_dict, batch_size=1)
            gold_temp_batch.non_tensor_batch = {
                key: (values[:1] if hasattr(values, '__len__') and len(values) >= 1 else values)
                for key, values in temp_batch.non_tensor_batch.items()
            }

            if 'is_gold_sequence_flags' in gold_temp_batch.non_tensor_batch:
                flags = gold_temp_batch.non_tensor_batch['is_gold_sequence_flags']
                if isinstance(flags, np.ndarray) and flags.size >= 1:
                    flags = flags.astype(object, copy=False)
                    flags[0] = True
                    gold_temp_batch.non_tensor_batch['is_gold_sequence_flags'] = flags
                else:
                    gold_temp_batch.non_tensor_batch['is_gold_sequence_flags'] = np.array([True], dtype=object)
            else:
                gold_temp_batch.non_tensor_batch['is_gold_sequence_flags'] = np.array([True], dtype=object)
            
            gold_scores = self.reward_fn(gold_temp_batch)
            gold_score = gold_scores.sum(-1).item()  # 스칼라
            
            # 3. 점수 비교 및 대체 결정
            max_generated_score = sequence_scores.max().item()
            min_generated_score = sequence_scores.min().item()
            
            if max_generated_score >= gold_score:
                continue
            
            # 4. 가장 낮은 점수 시퀀스를 골드로 대체
            min_score_idx = sequence_scores.argmin().item()
            actual_seq_idx = sequence_indices[min_score_idx]
            
            # 실제 골드 대체 수행
            self._replace_single_sequence(actual_seq_idx, group_idx, gold_tokens, responses, input_ids, 
                                        position_ids, attention_mask, prompts)
            
            # 골드 시퀀스 플래그 설정
            is_gold_sequence_flags[actual_seq_idx] = True
            gold_replacement_count += 1

        # 배치 업데이트
        batch.batch['responses'] = responses
        batch.batch['input_ids'] = input_ids
        batch.batch['position_ids'] = position_ids
        batch.batch['attention_mask'] = attention_mask
        
        # 업데이트된 플래그 배열을 다시 저장
        batch.non_tensor_batch['is_gold_sequence_flags'] = np.array(is_gold_sequence_flags, dtype=object)
        
        # 골드 대체 개수 정보를 배치에 저장 (나중에 서머리에서 사용)
        # non_tensor_batch 항목은 배치 크기와 동일한 길이의 np.array(object)를 기대하므로
        # 스칼라가 아닌 배열로 저장해야 reorder 시 인덱싱 오류가 나지 않는다.
        batch.non_tensor_batch['gold_replacement_count'] = np.full(batch_size, gold_replacement_count, dtype=object)
        
        return batch

    def _replace_single_sequence(self, seq_idx, gold_idx, gold_tokens, responses, input_ids, 
                               position_ids, attention_mask, prompts):
        """단일 시퀀스에 대한 골드 대체 로직"""
        # 1. responses 대체 - gold_idx를 사용해서 올바른 골드 토큰 가져오기
        responses[seq_idx] = gold_tokens[gold_idx]
        
        # 2. 새로운 input_ids 생성 (프롬프트 + 골드 응답)
        prompt_tokens = prompts[seq_idx]
        gold_resp = gold_tokens[gold_idx]
        
        # 프롬프트와 골드 응답 연결
        combined_tokens = torch.cat([prompt_tokens, gold_resp])
        
        # 길이 맞추기
        if combined_tokens.shape[0] > input_ids.shape[1]:
            # 너무 길면 자르기
            combined_tokens = combined_tokens[:input_ids.shape[1]]
        elif combined_tokens.shape[0] < input_ids.shape[1]:
            # 짧으면 패딩 추가
            pad_len = input_ids.shape[1] - combined_tokens.shape[0]
            pad_tokens = torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=combined_tokens.dtype)
            combined_tokens = torch.cat([combined_tokens, pad_tokens])
                    
        # 기존 1~2단계는 그대로 유지
        input_ids[seq_idx] = combined_tokens

        # 3. 새로운 attention_mask 생성 ― EOS(=pad)까지 1
        pad_id = self.tokenizer.pad_token_id
        non_pad_mask = (combined_tokens != pad_id)                 # pad 아닌 곳
        first_np = non_pad_mask.nonzero(as_tuple=True)[0][0].item()  # 첫 non-pad 인덱스
        tok_len  = non_pad_mask.sum().item()                         # 실제 토큰 수
        eos_idx  = min(first_np + tok_len, combined_tokens.size(0)-1)  # EOS 위치

        new_attention_mask = torch.zeros_like(combined_tokens, dtype=attention_mask.dtype)
        new_attention_mask[first_np : eos_idx + 1] = 1   # ← 좌패딩 건너뛰고 EOS까지 1
        attention_mask[seq_idx] = new_attention_mask

        # 4. 새로운 position_ids 생성 (pad는 0, 첫 토큰+1부터 1, 2, ...)
        first_np = (combined_tokens != pad_id).nonzero(as_tuple=True)[0][0]      # 첫 non-pad 인덱스
        new_position_ids = torch.arange(combined_tokens.size(0), dtype=position_ids.dtype)
        new_position_ids[:first_np + 1] = 0                   # 좌패딩 + 첫 토큰까지 0
        new_position_ids[first_np + 1:] -= first_np           # 두 번째 토큰부터 1부터 재정렬
        position_ids[seq_idx] = new_position_ids

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        import dill
        torch.save(self.train_dataloader, dataloader_local_path, pickle_module=dill)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        self.train_dataloader = torch.load(dataloader_local_path)
        if isinstance(self.train_dataloader.dataset, RLHFDataset):
            self.train_dataloader.dataset.resume_dataset_state()

        if self.config.trainer.get('override_lr_after_resume', False):
            new_actor_lr = self.config.actor_rollout_ref.actor.optim.lr
            print(f'Overriding actor lr after resume to {new_actor_lr}')
            self.actor_rollout_wg.override_lr(new_actor_lr)

            if self.use_critic:
                new_critic_lr = self.config.critic.optim.lr
                print(f'Overriding critic lr after resume to {new_critic_lr}')
                self.critic_wg.override_lr(new_critic_lr)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _write_otr_step_metrics(self, batch: DataProto):
        path = self.config.trainer.get('otr_step_metrics_path', None)
        if not path or not isinstance(path, str) or not path.strip():
            base_dir = self.config.trainer.get('otr_step_metrics_dir', None)
            if not base_dir or not isinstance(base_dir, str) or not base_dir.strip():
                return
            exp_name = str(self.config.trainer.experiment_name)
            safe_name = exp_name.replace(os.sep, "_")
            if os.altsep:
                safe_name = safe_name.replace(os.altsep, "_")
            path = os.path.join(base_dir.strip(), f"{safe_name}.jsonl")
        else:
            path = path.strip()

        def _as_list(val):
            if val is None:
                return []
            if isinstance(val, np.ndarray):
                return val.tolist()
            if isinstance(val, (list, tuple)):
                return list(val)
            return [val]

        def _to_int_list(vals, fill=0):
            out = []
            for v in vals:
                try:
                    out.append(int(v))
                except Exception:
                    out.append(fill)
            return out

        def _ensure_len(vals, target_len, fill=0):
            if len(vals) < target_len:
                return vals + [fill] * (target_len - len(vals))
            return vals[:target_len]

        try:
            group_ids = _as_list(batch.non_tensor_batch.get('uid') if 'uid' in batch.non_tensor_batch
                                 else batch.non_tensor_batch.get('otr_group_ids'))
            if not group_ids:
                return

            attempted_flags = [bool(v) for v in _as_list(batch.non_tensor_batch.get('otr_group_attempted_flags'))]
            replaced_flags = [bool(v) for v in _as_list(batch.non_tensor_batch.get('otr_replaced_flags'))]
            pre_output_tokens_per_seq = _to_int_list(_as_list(batch.non_tensor_batch.get('pre_output_tokens_per_seq')))
            otr_output_tokens_per_seq = _to_int_list(_as_list(batch.non_tensor_batch.get('otr_output_tokens_per_seq')))
            cut_output_tokens = _to_int_list(_as_list(batch.non_tensor_batch.get('otr_cut_output_tokens')))

            total_sequences = len(batch.batch['responses']) if 'responses' in batch.batch else len(group_ids)
            group_ids = group_ids[:total_sequences]
            pre_output_tokens_per_seq = _ensure_len(pre_output_tokens_per_seq, total_sequences, 0)
            otr_output_tokens_per_seq = _ensure_len(otr_output_tokens_per_seq, total_sequences, 0)
            cut_output_tokens = _ensure_len(cut_output_tokens, total_sequences, 0)

            pre_output_total = sum(pre_output_tokens_per_seq)
            pre_output_mean = (pre_output_total / total_sequences) if total_sequences > 0 else 0
            otr_output_total = sum(otr_output_tokens_per_seq)

            group_map = {}
            for i, gid in enumerate(group_ids):
                if gid is None:
                    continue
                group_map.setdefault(str(gid), []).append(i)

            group_entries = []
            resampled_group_ids = []
            replaced_group_ids = []

            for gid_key, indices in group_map.items():
                pre_tokens = [pre_output_tokens_per_seq[i] for i in indices]
                pre_total = sum(pre_tokens)
                pre_mean = (pre_total / len(pre_tokens)) if pre_tokens else 0
                attempted = any(i < len(attempted_flags) and attempted_flags[i] for i in indices)
                replaced = any(i < len(replaced_flags) and replaced_flags[i] for i in indices)
                if attempted:
                    resampled_group_ids.append(gid_key)
                if replaced:
                    replaced_group_ids.append(gid_key)

                group_entry = {
                    'group_id': gid_key,
                    'pre_output_tokens_per_seq': pre_tokens,
                    'pre_output_tokens_total': int(pre_total),
                    'pre_output_tokens_mean': pre_mean,
                    'was_resampled': attempted,
                    'was_replaced': replaced,
                }

                if attempted:
                    cut_val = 0
                    for idx in indices:
                        cut_val = int(cut_output_tokens[idx]) if idx < len(cut_output_tokens) else 0
                        break
                    otr_tokens = [otr_output_tokens_per_seq[i] for i in indices]
                    group_entry.update({
                        'cut_output_tokens': int(cut_val),
                        'otr_output_tokens_total': int(sum(otr_tokens)),
                        'otr_output_tokens_per_seq': otr_tokens,
                    })

                group_entries.append(group_entry)

            record = {
                'step': int(self.global_steps),
                'total_sequences': int(total_sequences),
                'rollout_n': int(self.config.actor_rollout_ref.rollout.n),
                'use_otr_sampling': bool(self.config.actor_rollout_ref.rollout.get('use_otr_sampling', False)),
                'resampled_group_count': int(len(resampled_group_ids)),
                'replaced_group_count': int(len(replaced_group_ids)),
                'resampled_group_ids': resampled_group_ids,
                'replaced_group_ids': replaced_group_ids,
                'pre_output_tokens_total': int(pre_output_total),
                'pre_output_tokens_mean': pre_output_mean,
                'otr_output_tokens_total': int(otr_output_total),
                'groups': group_entries,
            }

            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
        except Exception as e:
            print(f"[OTR] step metrics log failed: {type(e).__name__}: {e}")

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            # with open("/root/autodl-tmp/project/verl/full_content.txt", "a",encoding="utf-8") as f:
            #     f.write(f"val before train:\n")
            print(f"val before train:")
            val_metrics = self._validate()
            
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                # supporting_facts는 별도로 추가 (pop하지 않음)
                gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                
                # OTR sampling이 활성화된 경우 supporting_facts 및 ground_truth 별도 추가
                if self.config.actor_rollout_ref.rollout.get('use_otr_sampling', False):
                    if 'supporting_facts' in batch.non_tensor_batch:
                        gen_batch.non_tensor_batch['supporting_facts'] = batch.non_tensor_batch['supporting_facts']
                    
                    # ground_truth도 함께 전달 (점수 계산용)
                    if 'ground_truth' in batch.non_tensor_batch:
                        gen_batch.non_tensor_batch['ground_truth'] = batch.non_tensor_batch['ground_truth']
                        print(f"✅ Found ground_truth directly")
                    elif 'reward_model' in batch.non_tensor_batch:
                        # reward_model 안에서 ground_truth 추출
                        ground_truths = [rm.get('ground_truth', '') if isinstance(rm, dict) else '' 
                                       for rm in batch.non_tensor_batch['reward_model']]
                        gen_batch.non_tensor_batch['ground_truth'] = np.array(ground_truths, dtype=object)
                        print(f"✅ Found ground_truth in reward_model")
                    else:
                        print(f"⚠️ No ground_truth found. Available keys: {list(batch.non_tensor_batch.keys())}")

                with _timer('step', timing_raw):
                    # 검색 횟수 점진 상승 스케줄 적용 (옵션)
                    self._maybe_update_rollout_max_search()

                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    # OTR 사용 여부에 따라 다른 처리
                    use_otr_sampling = self.config.actor_rollout_ref.rollout.get('use_otr_sampling', False)
                    
                    if use_otr_sampling:
                        # OTR 모드: 기존 방식과 동일하게 처리 (배치 사이즈 고정)
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                        
                    else:
                        # 기존 고정 repeat 방식
                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
 
                    # 골드 시퀀스 대체 로직 
                    batch = self._replace_with_gold_sequence(batch)

                    responses = batch.batch['responses']
                    
                    response_length = responses.size(-1)
                    attention_mask = batch.batch['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    
                    if self.config.trainer.get('use_observation_mask', False):
                        loss_mask = self.identify_observation_mask(responses, response_mask)
                        batch.batch['loss_mask'] = loss_mask
                    
                    
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                                        
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty, 
                                                                 use_observation_mask=self.config.trainer.get('use_observation_mask', False))
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process

                        
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  use_observation_mask=self.config.trainer.get('use_observation_mask', False))
                                                  

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        # with open("/root/autodl-tmp/project/verl/full_content.txt", "a",encoding="utf-8") as f:
                        #     f.write(f"val at step {self.global_steps}:\n")
                        print(f"val at step {self.global_steps}:")
                        
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                
                # 배치 서머리 출력
                total_sequences = len(batch.batch['responses']) if 'responses' in batch.batch else 0
                gold_count = batch.non_tensor_batch.get('gold_replacement_count', 0)
                # gold_replacement_count는 배열로 저장되므로 요약에서는 스칼라만 사용
                if isinstance(gold_count, np.ndarray):
                    gold_count = gold_count[0] if gold_count.size > 0 else 0
                
                rollout_n = self.config.actor_rollout_ref.rollout.n
                use_otr_sampling = self.config.actor_rollout_ref.rollout.get('use_otr_sampling', False)
                
                # 평균 점수 계산
                if 'token_level_scores' in batch.batch:
                    scores = batch.batch['token_level_scores'].sum(-1)
                    avg_score = scores.mean().item()
                    
                    print(f"Batch Summary: {total_sequences} sequences, {gold_count} gold, avg score: {avg_score:.3f}")
                    
                    # 그룹별 분석 (OTR vs 일반 모드)
                    if use_otr_sampling and total_sequences > 0:
                        # 프롬프트별 시퀀스 개수 계산 및 디버깅 출력 (항상 그룹별 출력)
                        if 'uid' in batch.non_tensor_batch:
                            uid_groups = {}
                            uids = batch.non_tensor_batch['uid']
                            for i, uid in enumerate(uids):
                                if uid not in uid_groups:
                                    uid_groups[uid] = []
                                uid_groups[uid].append(i)
                            
                            # OTR 디버깅 플래그 읽기 (없을 수 있음)
                            otr_group_ids = batch.non_tensor_batch.get('otr_group_ids', None)
                            otr_resampled_flags = batch.non_tensor_batch.get('otr_group_resampled_flags', None)
                            otr_replaced_flags = batch.non_tensor_batch.get('otr_replaced_flags', None)
                            otr_plan_summary = batch.non_tensor_batch.get('otr_replacement_summary', {})
                            otr_plan_summary_dict = {}
                            if isinstance(otr_plan_summary, np.ndarray):
                                for entry in otr_plan_summary.tolist():
                                    if isinstance(entry, dict) and entry:
                                        pid = entry.get('prompt_id')
                                        if pid is not None:
                                            otr_plan_summary_dict[str(pid)] = entry
                            elif isinstance(otr_plan_summary, dict):
                                otr_plan_summary_dict = otr_plan_summary
                            else:
                                otr_plan_summary_dict = {}
                            
                            print(f"Groups ({len(uid_groups)} total), rollout.n={rollout_n}")
                            
                            for uid, indices in uid_groups.items():
                                group_scores = [scores[i].item() for i in indices]
                                max_score = max(group_scores) if group_scores else float('nan')
                                
                                # 그룹의 OTR 상태 추정: prompt_id 기반 플래그를 uid와 매핑하기 어렵다면, 해당 그룹 인덱스들의 flag에 OR
                                was_resampled = False
                                was_replaced = False
                                prompt_id_for_group = None
                                if otr_group_ids is not None:
                                    for i in indices:
                                        try:
                                            gid_raw = otr_group_ids[i]
                                            if gid_raw is None:
                                                continue
                                            prompt_id_for_group = int(gid_raw)
                                            break
                                        except Exception:
                                            continue
                                if otr_resampled_flags is not None and otr_replaced_flags is not None:
                                    for i in indices:
                                        try:
                                            if bool(otr_resampled_flags[i]):
                                                was_resampled = True
                                            if bool(otr_replaced_flags[i]):
                                                was_replaced = True
                                        except Exception:
                                            pass
                                
                                status = []
                                if was_resampled:
                                    status.append('resampled')
                                if was_replaced:
                                    status.append('replaced')
                                status_str = (', '.join(status)) if status else 'original'
                                
                                # 모든 그룹의 메타데이터 출력
                                if indices:
                                    first_idx = indices[0]
                                    try:
                                        # Supporting facts 출력
                                        sf_info = "(not found)"
                                        if 'supporting_facts' in batch.non_tensor_batch:
                                            sf_array = batch.non_tensor_batch['supporting_facts']
                                            if first_idx < len(sf_array):
                                                try:
                                                    sf = sf_array[first_idx]
                                                    if isinstance(sf, dict):
                                                        if 'supporting_facts' in sf:
                                                            sf_titles = sf['supporting_facts'].get('title', [])
                                                        else:
                                                            sf_titles = sf.get('title', [])
                                                        if sf_titles:
                                                            sf_info = f"{sf_titles}"
                                                        else:
                                                            sf_info = "(none)"
                                                    else:
                                                        sf_info = f"{sf}"
                                                except Exception as e:
                                                    sf_info = f"(indexing error: {e})"
                                        
                                        # Ground truth 출력 
                                        gt_info = "(not found)"
                                        if 'ground_truth' in batch.non_tensor_batch:
                                            gt_array = batch.non_tensor_batch['ground_truth']
                                            if first_idx < len(gt_array):
                                                try:
                                                    gt = gt_array[first_idx]
                                                    gt_info = f"{gt}"
                                                except Exception as e:
                                                    gt_info = f"(indexing error: {e})"
                                        elif 'reward_model' in batch.non_tensor_batch:
                                            rm_array = batch.non_tensor_batch['reward_model']
                                            if first_idx < len(rm_array):
                                                try:
                                                    rm = rm_array[first_idx]
                                                    if isinstance(rm, dict):
                                                        gt = rm.get('ground_truth', '(not found)')
                                                        gt_info = f"{gt}"
                                                except Exception as e:
                                                    gt_info = f"(rm indexing error: {e})"
                                    except Exception as e:
                                        sf_info = f"(error: {e})"
                                        gt_info = f"(error: {e})"
                                
                                if prompt_id_for_group is not None:
                                    print(f"   Group {uid} (prompt_id={prompt_id_for_group}): {len(indices)} sequences, max score: {max_score:.3f} [{status_str}]")
                                else:
                                    print(f"   Group {uid}: {len(indices)} sequences, max score: {max_score:.3f} [{status_str}]")
                                print(f"     SF: {sf_info}")
                                print(f"     GT: {gt_info}")
                                print(f"     Scores: {[f'{s:.2f}' for s in group_scores]}")
                                
                                if isinstance(otr_plan_summary_dict, dict):
                                    summary = None
                                    if prompt_id_for_group is not None:
                                        summary = otr_plan_summary_dict.get(str(prompt_id_for_group))
                                    if summary is None:
                                        summary = otr_plan_summary_dict.get(str(uid))
                                    if summary:
                                        picked_pairs = summary.get('picked_pairs', [])
                                        if picked_pairs:
                                            picked_str = [f"{pair['old']:.2f}->{pair['new']:.2f}" for pair in picked_pairs]
                                        else:
                                            picked_str = []
                                        print(f"     OTR picked: {picked_str} (sum {summary.get('sum_before', 0.0):.3f}->{summary.get('sum_after', 0.0):.3f})")
                                        skipped = summary.get('skipped_candidates', [])
                                        if skipped:
                                            skipped_str = [f"{item['score']:.2f}/{item['reason']}" for item in skipped]
                                            print(f"     OTR skipped: {skipped_str}")
                                
                    elif total_sequences > 0:
                        # 일반 모드: rollout_n 기반 그룹 분석
                        rollout_n = self.config.actor_rollout_ref.rollout.n
                        num_groups = total_sequences // rollout_n
                        
                        print(f"Groups ({num_groups} total), rollout.n={rollout_n}")
                        
                        # uid가 있으면 재정렬 이후에도 uid를 기준으로 그룹을 복원해서 보여준다.
                        uids = batch.non_tensor_batch.get('uid')
                        if isinstance(uids, np.ndarray):
                            uid_groups = {}
                            for i, uid in enumerate(uids):
                                uid_groups.setdefault(uid, []).append(i)
                            
                            print(f"   (uid 기반 그룹 복원: {len(uid_groups)}개)")
                            for group_idx, (uid, indices) in enumerate(uid_groups.items()):
                                group_scores = [scores[i].item() for i in indices]
                                max_score = max(group_scores) if group_scores else float('nan')
                                print(f"   Group {group_idx} (uid={uid}): {len(group_scores)} sequences, max score: {max_score:.3f}")
                                print(f"     Scores: {[f'{s:.2f}' for s in group_scores]}")
                        else:
                            for group_idx in range(num_groups):
                                group_start = group_idx * rollout_n
                                group_end = (group_idx + 1) * rollout_n
                                group_scores = [scores[i].item() for i in range(group_start, min(group_end, total_sequences))]
                                max_score = max(group_scores) if group_scores else float('nan')
                                
                                print(f"   Group {group_idx}: {len(group_scores)} sequences, max score: {max_score:.3f}")
                                print(f"     Scores: {[f'{s:.2f}' for s in group_scores]}")
                                



                # TODO: make a canonical logger that supports various backend
                self._write_otr_step_metrics(batch)
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return
