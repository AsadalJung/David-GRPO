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
Single Process Actor
"""

import itertools
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
from verl.workers.actor.loss_contract import encode_prompt_uids, prompt_balanced_loss_weights, weighted_token_sum
import verl.utils.torch_functional as verl_F

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

__all__ = ['DataParallelPPOActor']


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = torch.compile(verl_F.entropy_from_logits, dynamic=True)

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        # """
        # Returns: 
        #     entropy: # (bs, response_len)
        #     log_probs: # (bs, response_len)
        # """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        self.actor_optimizer.step()
        return grad_norm

    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        # """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        # Args:
        #     data (DataProto): a DataProto containing keys

        #         ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
        #         concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

        #         ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

        #         ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

        #         ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        # Returns:
        #     torch.Tensor: the log_prob tensor
        # """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        for micro_batch in micro_batches:
            with torch.no_grad():
                _, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
        log_probs = torch.concat(log_probs_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]

        return log_probs

    def _extract_prompt_indices(self, data: DataProto, micro_batch_size: int, batch_start_idx: int = 0):
        """Read canonical uid before TensorDict selection drops non-tensor metadata."""
        if 'uid' not in data.non_tensor_batch:
            raise ValueError('OTR prompt-balanced loss requires canonical uid metadata')
        ids = encode_prompt_uids(data.non_tensor_batch['uid'], data.batch.device)
        return ids[batch_start_idx:batch_start_idx + micro_batch_size]

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if self.config.get('use_action_loss_mask', True) and 'loss_mask' in data.batch:
            select_keys.append('loss_mask')
        batch = data.select(batch_keys=select_keys).batch
        balanced = self.config.get('otr_prompt_balanced_loss', False)
        if balanced:
            if 'loss_mask' not in batch:
                raise ValueError('OTR actor-loss contract requires the trainer action loss_mask')
            batch['prompt_ids'] = self._extract_prompt_indices(data, len(batch))

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        dataloader = batch.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for batch_idx, mini_batch in enumerate(dataloader):
            # split batch into micro_batches
            if balanced:
                width = mini_batch['responses'].size(-1)
                action_mask = mini_batch['loss_mask'] * mini_batch['attention_mask'][:, -width:]
                mini_batch['loss_weights'] = prompt_balanced_loss_weights(mini_batch['prompt_ids'], action_mask)
            mini_metrics = {}
            if self.config.use_dynamic_bsz:
                max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
            else:
                self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                # split batch into micro_batches
                micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

            self.actor_optimizer.zero_grad()

            for micro_batch_data in micro_batches:
                micro_batch_data = micro_batch_data.cuda()  # actor device is cpu when using offload
                responses = micro_batch_data['responses']
                response_length = responses.size(1)
                attention_mask = micro_batch_data['attention_mask']
                response_mask = attention_mask[:, -response_length:]
                if 'loss_mask' in micro_batch_data:
                    response_mask = response_mask * micro_batch_data['loss_mask']
                old_log_prob = micro_batch_data['old_log_probs']
                advantages = micro_batch_data['advantages']
                loss_weights = micro_batch_data.get('loss_weights', None)

                clip_ratio = self.config.clip_ratio
                entropy_coeff = self.config.entropy_coeff

                # all return: (bsz, response_length)
                entropy, log_prob = self._forward_micro_batch(micro_batch=micro_batch_data, temperature=temperature)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                              log_prob=log_prob,
                                                                              advantages=advantages,
                                                                              eos_mask=response_mask,
                                                                              cliprange=clip_ratio,
                                                                              loss_weights=loss_weights)
                # OTR 샘플링 균형 손실 적용
                if balanced:
                    entropy_loss = weighted_token_sum(entropy, response_mask, loss_weights)
                else:
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff

                if self.config.use_kl_loss:
                    ref_log_prob = micro_batch_data['ref_log_prob']
                    # compute kl loss
                    # Mask before nonlinear KL operations so excluded positions
                    # cannot inject an inf * 0 / NaN gradient during backward.
                    masked_log_prob = torch.where(response_mask.bool(), log_prob, torch.zeros_like(log_prob))
                    masked_ref_log_prob = torch.where(response_mask.bool(), ref_log_prob, torch.zeros_like(ref_log_prob))
                    kld = core_algos.kl_penalty(logprob=masked_log_prob,
                                                ref_logprob=masked_ref_log_prob,
                                                kl_penalty=self.config.kl_loss_type)
                    # KL 손실에도 균형 손실 적용
                    if balanced:
                        kl_loss = weighted_token_sum(kld, response_mask, loss_weights)
                    else:
                        kl_loss = masked_mean(kld, response_mask)

                    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                    if balanced:
                        mini_metrics['actor/kl_loss'] = mini_metrics.get('actor/kl_loss', 0.0) + kl_loss.detach().item()
                    else:
                        append_to_dict(metrics, {'actor/kl_loss': kl_loss.detach().item()})
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

                if balanced:
                    # Already normalized over the global optimizer minibatch.
                    loss = policy_loss
                elif self.config.use_dynamic_bsz:
                    # relative to the dynamic bsz
                    loss = policy_loss * (len(micro_batch_data) / self.config.ppo_mini_batch_size)
                else:
                    loss = policy_loss / self.gradient_accumulation
                loss.backward()

                micro_metrics = {
                    'actor/entropy_loss': entropy_loss.detach().item(),
                    'actor/pg_loss': pg_loss.detach().item(),
                    'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                    'actor/ppo_kl': ppo_kl.detach().item(),
                }
                if balanced:
                    for key, value in micro_metrics.items():
                        mini_metrics[key] = mini_metrics.get(key, 0.0) + value
                else:
                    append_to_dict(metrics, micro_metrics)

            if balanced:
                mini_metrics['actor/prompt_balanced_loss'] = 1.0
                mini_metrics['actor/action_token_fraction'] = (action_mask.sum() / mini_batch['attention_mask'][:, -width:].sum().clamp_min(1)).item()
                append_to_dict(metrics, mini_metrics)
                
            grad_norm = self._optimizer_step()
            data = {'actor/grad_norm': grad_norm.detach().item()}
            append_to_dict(metrics, data)
        self.actor_optimizer.zero_grad()
        return metrics
                
