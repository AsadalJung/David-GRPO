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

import math
from typing import List

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch


class NaiveRewardManager:
    """The reward manager.
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        give_partial_reward=False,
        use_answer_in_search_reward=False,
        require_search_match_for_answer=False,
        partial_reward_weight: float = 0.5,
        score_floor_threshold: float = 0.0,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.give_partial_reward = give_partial_reward
        self.use_answer_in_search_reward = use_answer_in_search_reward
        self.require_search_match_for_answer = require_search_match_for_answer
        try:
            weight_value = float(partial_reward_weight)
        except (TypeError, ValueError):
            weight_value = 0.5
        self.partial_reward_weight = min(max(weight_value, 0.0), 1.0)
        try:
            threshold_value = float(score_floor_threshold)
        except (TypeError, ValueError):
            threshold_value = 0.0
        self.score_floor_threshold = threshold_value if threshold_value > 0.0 else 0.0

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        batch_size = len(data)
        if batch_size == 0:
            if data.batch is not None and 'responses' in data.batch.keys():
                return torch.zeros_like(data.batch['responses'], dtype=torch.float32)
            return torch.zeros(0, dtype=torch.float32)

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            rm_scores = data.batch['rm_scores']
            if self.score_floor_threshold > 0.0:
                valid_response_lengths = self._extract_valid_response_lengths(data)
                seq_scores = rm_scores.sum(dim=-1, keepdim=False)
                numeric_scores = [self._safe_numeric(val) for val in seq_scores.detach().cpu().tolist()]
                adjusted_scores = self._apply_groupwise_floor(numeric_scores, data)
                rm_scores = self._scatter_scores_to_token_level(
                    adjusted_scores,
                    data=data,
                    valid_response_lengths=valid_response_lengths,
                    dtype=rm_scores.dtype,
                    device=rm_scores.device,
                )
            return rm_scores

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        raw_scores: List[float] = []
        batch_gold_flags: List[bool] = []
        valid_response_lengths: List[int] = []

        for i in range(batch_size):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            attention_mask = data_item.batch['attention_mask']
            valid_prompt_length_tensor = attention_mask[:prompt_length].sum()
            valid_prompt_length = int(valid_prompt_length_tensor.item())
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length_tensor = attention_mask[prompt_length:].sum()
            valid_response_length = int(valid_response_length_tensor.item())
            valid_response_lengths.append(valid_response_length)
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            # extra_info에 필요한 데이터만 포함
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            if not isinstance(extra_info, dict):
                extra_info = {}

            # 필요한 데이터만 extra_info에 추가 (메모리 절약)
            extra_info.update({
                'supporting_facts': data_item.non_tensor_batch.get('supporting_facts', {}),
                'prompt_length': valid_prompt_length,
                'response_length': valid_response_length,
                'data_source': data_source,
                'trainer_config': {
                    'give_partial_reward': getattr(self, 'give_partial_reward', False),
                    'use_answer_in_search_reward': getattr(self, 'use_answer_in_search_reward', False),
                    'require_search_match_for_answer': getattr(self, 'require_search_match_for_answer', False),
                    'partial_reward_weight': getattr(self, 'partial_reward_weight', 0.5),
                }
            })

            # 골드 시퀀스 감지: 배치 플래그 확인
            is_gold_sequence = False
            if hasattr(data, 'non_tensor_batch') and 'is_gold_sequence_flags' in data.non_tensor_batch:
                batch_flags = data.non_tensor_batch['is_gold_sequence_flags']
                try:
                    if i < len(batch_flags) and batch_flags[i]:
                        is_gold_sequence = True
                except (IndexError, TypeError):
                    pass

            if is_gold_sequence:
                raw_scores.append(1.0)
            else:
                score_raw = self.compute_score(
                    data_source=data_source,
                    solution_str=sequences_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                raw_scores.append(self._safe_numeric(score_raw))
            batch_gold_flags.append(is_gold_sequence)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)

        adjusted_scores = self._apply_groupwise_floor(raw_scores, data)

        tensor_width = reward_tensor.size(-1)
        for idx, (score, valid_len) in enumerate(zip(adjusted_scores, valid_response_lengths)):
            if valid_len <= 0:
                continue
            position = min(valid_len - 1, tensor_width - 1)
            reward_tensor[idx, position] = float(score)

        # 간단한 배치 요약만 출력
        if adjusted_scores:
            gold_count = sum(batch_gold_flags)
            avg_score = sum(adjusted_scores) / len(adjusted_scores)
            print(f"Batch Summary: {len(adjusted_scores)} sequences, {gold_count} gold, avg score: {avg_score:.3f}")

        return reward_tensor

    def _apply_score_floor(self, value):
        numeric = self._safe_numeric(value)
        if self.score_floor_threshold > 0.0 and numeric <= self.score_floor_threshold:
            return 0.0
        return numeric

    def _safe_numeric(self, value) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isnan(numeric) or math.isinf(numeric):
            return 0.0
        return numeric

    def _apply_groupwise_floor(self, scores: List[float], data: DataProto) -> List[float]:
        if not scores:
            return []

        numeric_scores = [self._safe_numeric(score) for score in scores]
        if self.score_floor_threshold <= 0.0:
            return numeric_scores

        adjusted_scores = list(numeric_scores)
        groups = self._extract_groups(data, len(numeric_scores))

        for group in groups:
            if not group:
                continue
            group_values = [numeric_scores[idx] for idx in group]
            clean_values = [val for val in group_values if not math.isnan(val)]
            if not clean_values:
                continue
            if max(clean_values) <= self.score_floor_threshold:
                for idx in group:
                    adjusted_scores[idx] = 0.0

        return adjusted_scores

    def _extract_groups(self, data: DataProto, total: int) -> List[List[int]]:
        if total <= 0:
            return []

        for key in ('uid', 'otr_group_ids'):
            arr = data.non_tensor_batch.get(key)
            if arr is None:
                continue

            try:
                if len(arr) != total:
                    continue
            except TypeError:
                continue

            group_map = {}
            for idx in range(total):
                group_key = self._normalize_group_key(arr[idx])
                group_map.setdefault(group_key, []).append(idx)

            if group_map:
                return list(group_map.values())

        return [[idx] for idx in range(total)]

    @staticmethod
    def _normalize_group_key(value):
        if isinstance(value, bytes):
            try:
                return value.decode('utf-8')
            except Exception:
                return value
        if hasattr(value, 'item') and not isinstance(value, (str, bytes)):
            try:
                return value.item()
            except Exception:
                return value
        return value

    def _extract_valid_response_lengths(self, data: DataProto) -> List[int]:
        if data.batch is None or \
                'responses' not in data.batch.keys() or \
                'attention_mask' not in data.batch.keys():
            return [0] * len(data)

        responses = data.batch['responses']
        attention_mask = data.batch['attention_mask']
        response_length = responses.shape[-1]
        response_mask = attention_mask[:, -response_length:]
        lengths = response_mask.sum(dim=-1)
        return [int(length.item()) for length in lengths]

    def _scatter_scores_to_token_level(
        self,
        scores: List[float],
        data: DataProto,
        valid_response_lengths: List[int],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        response_length = data.batch['responses'].shape[-1]
        token_level = torch.zeros((len(scores), response_length), dtype=dtype, device=device)
        for idx, (score, valid_len) in enumerate(zip(scores, valid_response_lengths)):
            if valid_len <= 0:
                continue
            position = min(valid_len - 1, response_length - 1)
            token_level[idx, position] = float(score)
        return token_level
