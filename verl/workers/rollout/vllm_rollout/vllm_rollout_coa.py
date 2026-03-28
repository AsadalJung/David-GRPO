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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
from typing import List, Dict, Any
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from tqdm import tqdm

from verl import DataProto
from verl.utils.torch_functional import get_eos_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.third_party.vllm import LLM, vllm_version
from verl.third_party.vllm import parallel_state as vllm_ps
from vllm import SamplingParams

import copy
import os

import re

import requests
import json
import time
import asyncio
import httpx

# Retriever endpoint can be overridden via RETRIEVER_URL env var.
BASE_URL = os.getenv("RETRIEVER_URL", "http://localhost:8001/retrieve")
_LEGACY_VLLM_VERSIONS = {"0.3.1", "0.4.2", "0.5.4", "0.6.3"}


def _uses_legacy_vllm_api() -> bool:
    return vllm_version in _LEGACY_VLLM_VERSIONS


def _extract_output_token_ids(output) -> List[List[int]]:
    if _uses_legacy_vllm_api():
        return output[0].tolist()

    token_ids: List[List[int]] = []
    for request_output in output:
        for sample_output in request_output.outputs:
            token_ids.append(sample_output.token_ids)
    return token_ids


def _normalize_eos_token_ids(eos_token_id):
    """Return (primary_id, [ids...]) even when the config exposes multiple EOS ids."""
    if eos_token_id is None:
        return None, []
    if isinstance(eos_token_id, torch.Tensor):
        eos_token_id = eos_token_id.tolist()
    if isinstance(eos_token_id, (list, tuple)):
        normalized = []
        for token in eos_token_id:
            if isinstance(token, (list, tuple)):
                normalized.extend(token)
            else:
                normalized.append(int(token))
        primary = normalized[0] if normalized else None
        return primary, normalized
    return int(eos_token_id), [int(eos_token_id)]

# Determine whether to use Tree-GRPO retriever (port 8003) or AutoCoA retriever (default)
def _is_tree_retriever():
    try:
        port = BASE_URL.split(":")[-1].split("/")[0]
        return port in {"8003", "8004"}
    except Exception:
        return False


def do_retrevial(text, top_k=3, return_score=True):
    """
    Single query retrieval with two protocol variants:
    - AutoCoA retriever (default): payload {"query": text, "tok_k": top_k, "return_score": return_score}
    - Tree retriever (port 8003): payload {"queries": [text], "topk": top_k, "return_scores": True}
      (Tree server has a bug on return_scores=False, so keep True)
    """
    tree_mode = _is_tree_retriever()
    if tree_mode:
        payload = {"queries": [text], "topk": top_k, "return_scores": True}
    else:
        payload = {"query": text, "tok_k": top_k, "return_score": return_score}

    try:
        response = requests.post(BASE_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        if tree_mode:
            # expect {"result": [[{"document": {...}, "score": ...}, ...]]}
            return data.get("result", [])
        else:
            # expect {"documents": [...], "scores": [...]}
            return data
    except requests.exceptions.RequestException as e:
        print(f"Single query failed: {e}")
        return None


def do_batch_retrevial(text_list, top_k=3, return_score=True, batch_size=512):
    """
    Batch retrieval with two protocol variants (see do_retrevial docstring).
    """
    tree_mode = _is_tree_retriever()
    if tree_mode:
        payload = {"queries": text_list, "topk": top_k, "return_scores": True}
    else:
        payload = {"query": text_list, "tok_k": top_k, "return_score": return_score}

    retries = 2
    while retries > 0:
        try:
            response = requests.post(BASE_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            if tree_mode:
                # data: {"result": [[{"document": {...}, "score": ...}, ...], ...]}
                return data.get("result", [])
            else:
                # data: list of dict when payload query is list
                return data if isinstance(data, list) else [data]
        except requests.exceptions.RequestException as e:
            print(f"Batch query failed: {e}")
            retries -= 1
            if retries == 0:
                return [None] * len(text_list)
            continue


def topk_format(result, topk=1):
    """
    Normalize retrieval results into a string. Supports both protocols.
    """
    tree_mode = _is_tree_retriever()
    if tree_mode:
        # result is a list of {"document": {...}, "score": ...}
        documents = [item["document"] for item in result]
    else:
        documents = result["documents"]

    if topk == 1:
        return documents[0]["contents"]
    else:
        content_list = []
        for index, doc in enumerate(documents, start=1):
            content_list.append(f"result {index}: {doc['contents']}")
        return "\n".join(content_list)
def get_search_results(texts):
    
    # search_reslut_token = "<search_result> {search_result} </search_result>"
    

    retrevial_result = do_batch_retrevial(texts, top_k=3)
    
    search_result_list = []
    for res in retrevial_result:
        if res is None:
            # search_result_list.append("Retrevial failed.")
            search_reslut_token = "<search_result> Retrevial failed. </search_result>\n\n"
            search_result_list.append(search_reslut_token)
        else:
            # search_result_list.append(topk_format(res, topk=3))
            search_reslut_token = "<search_result> " + topk_format(res, topk=3) + " </search_result>\n\n"
            search_result_list.append(search_reslut_token)
            
    
        
    return search_result_list

    
# Format the generated text
def check_search_token(text):
    pattern = r"<begin_search>(.*?)</end_search>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip(), text[:match.end()]
    return "", ""


def normalize_title_for_otr(title):
    """OTR(Optimal Truncation Resampling) sampling용 타이틀 정규화 (hotpotqa.py와 동일)"""
    if not title:
        return ""
    
    # HTML 엔티티 디코딩
    html_entities = {
        '&amp;': '&',
        '&lt;': '<',
        '&gt;': '>',
        '&quot;': '"',
        '&#39;': "'",
        '&apos;': "'",
        '&nbsp;': ' ',
        '&copy;': '©',
        '&reg;': '®',
        '&trade;': '™'
    }
    
    for entity, char in html_entities.items():
        title = title.replace(entity, char)
    
    # 기본 정규화 (소문자, 공백 정리)
    normalized = title.lower().strip()
    
    # 특수 문자 제거 (하지만 공백은 유지)
    normalized = re.sub(r'[^\w\s]', '', normalized)
    
    # 연속된 공백을 하나로
    normalized = re.sub(r'\s+', ' ', normalized)
    
    return normalized.strip()


def extract_titles_from_search_results_otr(solution_str):
    """검색 결과에서 타이틀들을 추출합니다 (hotpotqa.py와 동일).
    
    Args:
        solution_str: 모델의 응답 텍스트
        
    Returns:
        list: 추출된 타이틀들의 리스트
    """
    titles = []
    
    # <search_result> 태그들을 찾습니다
    search_result_pattern = r'<search_result>(.*?)</search_result>'
    search_results = re.findall(search_result_pattern, solution_str, re.DOTALL)
    
    for result in search_results:
        # result 1: "Title" 형태를 찾습니다
        title_pattern = r'result\s+\d+:\s*"([^"]+)"'
        title_matches = re.findall(title_pattern, result)
        titles.extend(title_matches)
    
    return titles


# hotpotqa.py에서 점수 계산 함수들 import
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utils/reward_score'))
    from hotpotqa import compute_score
    HOTPOTQA_AVAILABLE = True
    # hotpotqa module loaded successfully
except ImportError as e:
    # Warning: hotpotqa module not found
    HOTPOTQA_AVAILABLE = False


def score_sequences_simple(full_sequences, ground_truths, supporting_facts_list):
    """간단한 시퀀스 점수 계산 함수"""
    if not HOTPOTQA_AVAILABLE:
        # Skipping sequence scoring - hotpotqa module not available
        return []
    
    all_scores = []
    
    for i, sequence in enumerate(full_sequences):
        # ground_truth 추출
        if i < len(ground_truths):
            ground_truth = ground_truths[i]
        else:
            ground_truth = ground_truths[0] if ground_truths else ""
        
        # supporting_facts 추출
        if i < len(supporting_facts_list):
            facts_meta = supporting_facts_list[i]
            if isinstance(facts_meta, dict):
                supporting_facts = facts_meta.get('supporting_facts', {})
            else:
                supporting_facts = facts_meta if facts_meta else {}
        else:
            supporting_facts = {}
        
        try:
            # extra_info 구성
            extra_info = {
                'supporting_facts': supporting_facts,
                'trainer_config': {
                    'give_partial_reward': True,
                    'use_answer_in_search_reward': self.config.get('use_answer_in_search_reward', False),
                    'require_search_match_for_answer': self.config.get('require_search_match_for_answer', False),
                    'partial_reward_weight': self.config.get('partial_reward_weight', 0.5),
                }
            }
            
            # hotpotqa 점수 계산
            score = compute_score(
                solution_str=sequence,
                ground_truth=ground_truth,
                method='strict',
                format_score=0.5,
                score=1.0,
                extra_info=extra_info
            )
            
            all_scores.append({
                'sequence_idx': i,
                'total_score': score,
                'has_think_tag': '</think>' in sequence,
                'has_search': '<begin_search>' in sequence,
                'sequence_length': len(sequence)
            })
            
        except Exception as e:
            # Error scoring sequence
            all_scores.append({
                'sequence_idx': i,
                'total_score': 0.0,
                'has_think_tag': False,
                'has_search': False,
                'sequence_length': len(sequence),
                'error': str(e)
            })
    
    return all_scores


def count_searches_in_response_only(full_sequence, prompt_length):
    """응답 부분에서만 검색 횟수 카운트 (프롬프트 제외)"""
    if len(full_sequence) <= prompt_length:
        return 0
    
    # 응답 부분만 추출
    response_part = full_sequence[prompt_length:]
    return len(re.findall(r'<search_result>.*?</search_result>', response_part, re.DOTALL))


def count_searches_in_sequence(sequence):
    """전체 시퀀스에서 검색 횟수 카운트 (호환성을 위해 유지)"""
    return len(re.findall(r'<search_result>.*?</search_result>', sequence, re.DOTALL))


def get_remaining_searches(sequence, max_search_nums, prompt_length=None):
    """남은 검색 횟수 계산"""
    if prompt_length is not None:
        # 프롬프트 길이가 주어진 경우 응답 부분만 카운트
        used = count_searches_in_response_only(sequence, prompt_length)
    else:
        # 기존 방식 (전체 시퀀스에서 카운트)
        used = count_searches_in_sequence(sequence)
    return max(0, max_search_nums - used)


def analyze_supporting_match_completeness(sequence, supporting_facts):
    """Supporting facts 매칭 완성도 분석"""
    if not supporting_facts or 'title' not in supporting_facts:
        return {
            'total_required': 0,
            'unique_found': 0,
            'is_complete': True,  # supporting facts가 없으면 완성된 것으로 간주
            'match_titles': []
        }
    
    required_titles = supporting_facts['title']
    if isinstance(required_titles, str):
        required_titles = [required_titles]
    
    # 시퀀스에서 찾은 타이틀들
    found_titles = extract_titles_from_search_results_otr(sequence)
    
    # 정규화
    normalized_required = list(set([normalize_title_for_otr(t) for t in required_titles]))
    normalized_found = [normalize_title_for_otr(t) for t in found_titles]
    
    # 매칭된 고유 타이틀들
    unique_matches = list(set(normalized_found) & set(normalized_required))
    
    return {
        'total_required': len(normalized_required),
        'unique_found': len(unique_matches),
        'is_complete': len(unique_matches) >= len(normalized_required),
        'match_titles': unique_matches
    }


def find_optimal_truncation_point(sequence, supporting_facts, prompt_length, max_redundant=3, debug=False):
    """최적 절단점 찾기 - Supporting facts 완성 후 적절한 지점에서 절단"""
    
    # 매칭 분석은 반드시 응답 구간(프롬프트 이후)만 대상으로 수행
    response_only = sequence[prompt_length:]
    match_analysis = analyze_supporting_match_completeness(response_only, supporting_facts)
    
    if not match_analysis['is_complete']:
        # Supporting facts가 완성되지 않았거나 매칭이 없는 경우
        if match_analysis['total_required'] > 0 and match_analysis['unique_found'] == 0:
            # 매칭되는 결과가 하나도 없으면 처음부터 다시 생성
            return prompt_length
        else:
            # 부분적으로 매칭되었으면 마지막 매칭 지점에서 절단 시도
            pass
    
    if match_analysis['total_required'] == 0:
        return len(sequence)  # supporting facts가 없으면 전체 유지
    
    # 검색 결과들 찾기 (트레일링 개행에 의존하지 않도록 보다 관대하게 매칭)
    search_results = list(re.finditer(r'<search_result>([\s\S]*?)</search_result>', sequence, re.DOTALL))
    
    required_titles = supporting_facts.get('title', [])
    if isinstance(required_titles, str):
        required_titles = [required_titles]
    normalized_required = set([normalize_title_for_otr(t) for t in required_titles])
    
    # 규칙: 시퀀스 내에서 '고유 매칭 수'의 최대값을 처음 달성한 지점에서 절단
    seen_required = set()
    max_unique_count = 0
    earliest_cut_for_max = None
    
    for match in search_results:
        if match.start() < prompt_length:
            continue
        
        # 전체 블록을 전달하여 파서가 포맷 차이에 덜 민감하도록 함
        result_text = match.group(0)
        found_titles = extract_titles_from_search_results_otr(result_text)
        
        updated = False
        for title in found_titles:
            normalized = normalize_title_for_otr(title)
            if normalized in normalized_required and normalized not in seen_required:
                seen_required.add(normalized)
                updated = True
        
        current_count = len(seen_required)
        if current_count > max_unique_count:
            max_unique_count = current_count
            earliest_cut_for_max = match.end()
    
    # 최종 절단점 결정
    if max_unique_count == 0:
        return prompt_length
    else:
        # </search_result> 이후 \n\n 경계까지 정렬
        def _align_cut_to_double_newline(seq, base_idx):
            if base_idx is None:
                return None
            # 즉시 두 개 개행이면 그대로 +2
            if seq[base_idx:base_idx+2] == "\n\n":
                return base_idx + 2
            # 근처(최대 32자)에서 두 개 개행을 찾으면 거기로 정렬
            next_dbl = seq.find("\n\n", base_idx)
            if next_dbl != -1 and next_dbl - base_idx <= 32:
                return next_dbl + 2
            # 기본값: 정렬하지 않고 base 반환
            return base_idx
        aligned_cut = _align_cut_to_double_newline(sequence, earliest_cut_for_max)
        return aligned_cut


def new_otr_resampling_logic(full_sequences, supporting_facts_list, ground_truths, prompt_lengths, score_threshold=1.0, rollout_n=5, max_search_nums=10):
    """OTR(Optimal Truncation Resampling) - 그룹 내 최고 점수가 1 미만일 때만 재샘플링.

    새 전략: 그룹에서 가장 점수가 높은 시퀀스 하나를 잘라 복제한 뒤, 그룹 크기만큼 재생성하여
    최고 후보가 기존 최고 점수를 넘어설 경우 해당 후보를 그룹 내 최저 점수 시퀀스와 교체한다.
    """

    all_scores = score_sequences_simple(full_sequences, ground_truths, supporting_facts_list)
    if not all_scores:
        return [], full_sequences, supporting_facts_list

    prompt_groups = {}
    for i, facts_meta in enumerate(supporting_facts_list):
        if isinstance(facts_meta, dict) and 'original_prompt_id' in facts_meta:
            prompt_id = facts_meta['original_prompt_id']
        else:
            raise ValueError(f"Missing original_prompt_id in supporting_facts_list at index {i}: {facts_meta}")

        prompt_groups.setdefault(prompt_id, []).append(i)

    groups_to_resample = []
    final_sequences = full_sequences.copy()
    final_supporting_facts = supporting_facts_list.copy()

    for prompt_id, sequence_indices in prompt_groups.items():
        if not sequence_indices:
            continue

        group_scores = []
        for seq_idx in sequence_indices:
            if seq_idx < len(all_scores):
                score = all_scores[seq_idx].get('total_score', 0.0)
            else:
                score = 0.0
            group_scores.append((seq_idx, score))

        max_score = max(score for _, score in group_scores)
        if max_score >= score_threshold:
            continue

        all_zero_group = all(score == 0.0 for _, score in group_scores)
        score_map = {seq_idx: score for seq_idx, score in group_scores}

        best_seq_idx, best_score = max(group_scores, key=lambda item: item[1])
        worst_seq_idx, worst_score = min(group_scores, key=lambda item: item[1])

        best_sequence = full_sequences[best_seq_idx]
        best_sf_meta = supporting_facts_list[best_seq_idx] if best_seq_idx < len(supporting_facts_list) else {}
        if isinstance(best_sf_meta, dict) and 'supporting_facts' in best_sf_meta:
            best_supporting_payload = best_sf_meta['supporting_facts']
        else:
            best_supporting_payload = best_sf_meta if best_sf_meta else {}

        best_prompt_length = prompt_lengths[best_seq_idx] if best_seq_idx < len(prompt_lengths) else 0
        cut_point = find_optimal_truncation_point(best_sequence, best_supporting_payload, best_prompt_length, debug=False)
        cut_offset = max(0, int(cut_point) - int(best_prompt_length))
        truncated_sequence = best_sequence[:cut_point]
        remaining_searches = get_remaining_searches(truncated_sequence, max_search_nums, best_prompt_length)
        best_ground_truth = ground_truths[best_seq_idx] if best_seq_idx < len(ground_truths) else ""

        default_ground_truth = ground_truths[sequence_indices[0]] if sequence_indices and sequence_indices[0] < len(ground_truths) else ""
        default_supporting_meta = supporting_facts_list[sequence_indices[0]] if sequence_indices else {}

        groups_to_resample.append({
            'prompt_id': prompt_id,
            'original_indices': sequence_indices,
            'best_sequence_idx': best_seq_idx,
            'worst_sequence_idx': worst_seq_idx,
            'best_score': float(best_score),
            'worst_score': float(worst_score),
            'best_sequence': {
                'seq_idx': best_seq_idx,
                'original_sequence': best_sequence,
                'original_score': float(best_score),
                'supporting_facts_meta': best_sf_meta,
                'supporting_facts_payload': best_supporting_payload if best_supporting_payload else {},
                'ground_truth': best_ground_truth,
                'truncated_prompt': truncated_sequence,
                'remaining_searches': remaining_searches,
                'prompt_length': best_prompt_length
            },
            'max_score': max_score,
            'all_zero': all_zero_group,
            'cut_offset': cut_offset,
            'default_ground_truth': default_ground_truth,
            'default_supporting_facts': default_supporting_meta,
            'original_scores': [score_map.get(seq_idx, 0.0) for seq_idx in sequence_indices]
        })

    return groups_to_resample, final_sequences, final_supporting_facts


def process_search_answer_batch(ans_list, current_prefix_list, reach_limit=False):
    search_queries = []
    ans_modified_list = []
    for ans in ans_list:
        search_query, ans_modified = check_search_token(ans.split("</think>")[0])
        search_queries.append(search_query)
        ans_modified_list.append(ans_modified)
    
    new_prefix_list = [None] * len(current_prefix_list)  
    search_flag_list = [False] * len(current_prefix_list)  
    
    for i, search_query in enumerate(search_queries):
        if search_query == "":
            new_prefix_list[i] = current_prefix_list[i]  
            search_flag_list[i] = False
        else:
            if reach_limit:
                new_prefix = current_prefix_list[i] + f"{ans_modified_list[i]}\n\n<search_result> Reach the limit of search times. </search_result>\n\n"
                new_prefix_list[i] = new_prefix  
                search_flag_list[i] = True
            else:
                search_flag_list[i] = True
    
    if not reach_limit and any(search_flag_list):
        try:
            queries_to_search = []
            query_indices = []  
            for i, flag in enumerate(search_flag_list):
                if flag:
                    queries_to_search.append(search_queries[i])
                    query_indices.append(i)
            
            search_results = get_search_results(queries_to_search)
            
            assert len(search_results) == len(queries_to_search), f"검색 결과 수와 쿼리 수가 일치하지 않습니다: {len(search_results)} vs {len(queries_to_search)}"
            
            for idx, result_idx in enumerate(query_indices):
                if idx < len(search_results):  
                    new_prefix = current_prefix_list[result_idx] + f"{ans_modified_list[result_idx]}\n\n{search_results[idx]}\n\n"
                    new_prefix_list[result_idx] = new_prefix  
        
        except Exception as e:
            print("An error occurred during the search process: ", e)
            for i in range(len(search_queries)):
                if search_flag_list[i] and new_prefix_list[i] is None:
                    new_prefix = current_prefix_list[i] + f"{ans_modified_list[i]}\n\n<search_result> Retrieval failed due to an error. </search_result>\n\n"
                    new_prefix_list[i] = new_prefix
    
    return new_prefix_list, search_flag_list
    

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


def extract_prompt_sequence_counts(supporting_facts_list, original_batch_size, total_sequences):
    """supporting_facts_list에서 각 프롬프트별 시퀀스 개수를 추출하는 공통 함수
    
    Returns:
        list: 각 프롬프트별 시퀀스 개수 [prompt0_count, prompt1_count, ...]
    """
    sequences_count = [0] * original_batch_size
    
    # supporting_facts_list에서 원본 프롬프트 ID 추출
    for i, facts_meta in enumerate(supporting_facts_list):
        if i >= total_sequences:  # 최종 시퀀스 수 초과하면 중단
            break
            
        if isinstance(facts_meta, dict) and 'original_prompt_id' in facts_meta:
            prompt_id = facts_meta['original_prompt_id']
            if prompt_id < len(sequences_count):
                sequences_count[prompt_id] += 1
        else:
            # fallback: 균등 분배 추정
            prompt_id = i % original_batch_size
            sequences_count[prompt_id] += 1
    
    # 누락된 시퀀스들은 균등 분배로 추정
    remaining_sequences = total_sequences - len(supporting_facts_list)
    if remaining_sequences > 0:
        sequences_per_prompt = remaining_sequences // original_batch_size
        extra_sequences = remaining_sequences % original_batch_size
        for prompt_id in range(original_batch_size):
            additional = sequences_per_prompt + (1 if prompt_id < extra_sequences else 0)
            sequences_count[prompt_id] += additional
    
    return sequences_count


class vLLMRollout(BaseRollout):

    def __init__(self,
                 actor_module: nn.Module = None,
                 config: DictConfig = None,
                 tokenizer=None,
                 model_hf_config=None,
                 model_path: str = None,
                 **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config
        assert not (not config.enforce_eager and config.free_cache_engine), \
            "disable CUDA graph (enforce_eager = False) if free cache engine"

        tensor_parallel_size = self.config.get('tensor_model_parallel_size', 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), \
            "tensor parallel size should be less than or equal to the world size"
        max_num_batched_tokens = self.config.get('max_num_batched_tokens', 8192)

        if kwargs.get('train_tp', None) is not None:
            # deployed with megatron
            import os
            os.environ['CUDA_TIMER_STREAM_KAFKA_ENABLE'] = '0'
            os.environ['MEGATRON_IMPORT_TIMERS'] = '0'
            train_tp = kwargs.get('train_tp', None)
            num_tp_per_train_tp = train_tp // tensor_parallel_size
            if _uses_legacy_vllm_api():
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)
            else:
                vllm_ps.initialize_parallel_state(tensor_model_parallel_size=tensor_parallel_size,
                                                  num_tp_per_train_tp=num_tp_per_train_tp)

        assert model_hf_config.max_position_embeddings >= config.prompt_length + config.response_length, \
            "model context length should be greater than total sequence length"
        if _uses_legacy_vllm_api():
            self.inference_engine = LLM(
                actor_module,
                tokenizer=tokenizer,
                model_hf_config=model_hf_config,
                tensor_parallel_size=tensor_parallel_size,
                seed=config.get('seed', 0),
                dtype=config.dtype,
                enforce_eager=config.enforce_eager,
                gpu_memory_utilization=config.gpu_memory_utilization,
                skip_tokenizer_init=False,
                max_model_len=config.prompt_length + config.response_length,
                load_format=config.load_format,
                disable_log_stats=config.disable_log_stats,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=config.enable_chunked_prefill,
            )
            self.inference_engine.offload_model_weights()
        else:
            if model_path is None:
                raise ValueError("model_path is required for search rollout with newer vLLM versions")
            self.inference_engine = LLM(
                model=model_path,
                enable_sleep_mode=True,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend="external_launcher",
                seed=config.get('seed', 0),
                dtype=config.dtype,
                enforce_eager=config.enforce_eager,
                gpu_memory_utilization=config.gpu_memory_utilization,
                disable_custom_all_reduce=True,
                skip_tokenizer_init=False,
                max_model_len=config.prompt_length + config.response_length,
                disable_log_stats=config.disable_log_stats,
                max_num_batched_tokens=max_num_batched_tokens,
                enable_chunked_prefill=config.enable_chunked_prefill,
            )
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=1,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
            stop=["</end_search>"],  # stop generation at end_search token
            include_stop_str_in_output=True,  # include stop string in output
        )

        # Stop strings require detokenization in both legacy and newer APIs.
        kwargs['detokenize'] = True

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)

        self.sampling_params = SamplingParams(**kwargs)

        self.pad_token_id = tokenizer.pad_token_id

        self.tokenizer = tokenizer
        
        self.max_search_nums = config.get('max_search_nums', 10)
        self.use_otr_sampling = config.get('use_otr_sampling', False)  # Enable OTR (Optimal Truncation Resampling) mode when set.
        self.tensor_parallel_size = tensor_parallel_size  # DataProto chunk에서 사용
        
        if self.use_otr_sampling:
            pass  # OTR Sampling enabled

    def set_max_search_nums(self, max_search_nums: int):
        """동적으로 최대 검색 횟수를 조정한다."""
        try:
            max_search_nums = int(max_search_nums)
        except Exception:
            return
        self.max_search_nums = max(0, max_search_nums)

    def _run_batch_resample_generation(self, truncated_input_ids: List[List[int]], remaining_searches: List[int]) -> tuple[List[str], List[str], Dict[str, int], List[int]]:
        """truncated 프롬프트를 재검색/재생성하여 prefix(프롬프트)와 response(응답)를 반환한다."""
        if not truncated_input_ids:
            return [], [], {'prompt_tokens': 0, 'response_tokens': 0}, []

        batch_resample_params = copy.deepcopy(self.sampling_params)
        batch_resample_params.n = 1
        batch_resample_params.max_tokens = 2048

        initial_output = self.inference_engine.generate(
            prompts=None,
            sampling_params=batch_resample_params,
            prompt_token_ids=truncated_input_ids,
            use_tqdm=False
        )

        response_token_ids = initial_output[0].tolist()
        resample_prompt_tokens = sum(len(ids) for ids in truncated_input_ids)
        resample_response_tokens = sum(len(ids) for ids in response_token_ids)
        per_seq_output_tokens = [len(ids) for ids in response_token_ids]
        current_response_strs = self.tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)
        current_prefix_list = [
            self.tokenizer.decode(prompt_ids, skip_special_tokens=False) for prompt_ids in truncated_input_ids
        ]

        used_searches = [0] * len(truncated_input_ids)
        max_searches = max(remaining_searches) if remaining_searches else 0
        single_resample_params = copy.deepcopy(batch_resample_params)
        single_resample_params.max_tokens = 1024

        for search_iter in range(max_searches):
            sequences_reach_limit = []
            for used, remaining in zip(used_searches, remaining_searches):
                reach_limit = (search_iter >= remaining - 1) or (used >= remaining)
                sequences_reach_limit.append(reach_limit)

            new_prefix_list, search_flag_list = self._process_search_answer_batch_with_individual_limits(
                current_response_strs,
                current_prefix_list,
                sequences_reach_limit
            )

            current_prefix_list = new_prefix_list

            active_indices = []
            for idx, flag in enumerate(search_flag_list):
                if flag and not sequences_reach_limit[idx]:
                    used_searches[idx] += 1
                    active_indices.append(idx)

            if not active_indices:
                break

            active_token_ids = [
                self.tokenizer.encode(current_prefix_list[idx], add_special_tokens=False) for idx in active_indices
            ]

            if not active_token_ids:
                break

            output = self.inference_engine.generate(
                prompts=None,
                sampling_params=single_resample_params,
                prompt_token_ids=active_token_ids,
                use_tqdm=False
            )

            new_response_token_ids = output[0].tolist()
            resample_prompt_tokens += sum(len(ids) for ids in active_token_ids)
            resample_response_tokens += sum(len(ids) for ids in new_response_token_ids)
            new_response_strs = self.tokenizer.batch_decode(new_response_token_ids, skip_special_tokens=False)

            for idx, seq_idx in enumerate(active_indices):
                if idx < len(new_response_strs):
                    current_response_strs[seq_idx] = new_response_strs[idx]
                if idx < len(new_response_token_ids) and seq_idx < len(per_seq_output_tokens):
                    per_seq_output_tokens[seq_idx] += len(new_response_token_ids[idx])

        token_stats = {
            'prompt_tokens': int(resample_prompt_tokens),
            'response_tokens': int(resample_response_tokens),
        }
        return current_prefix_list, current_response_strs, token_stats, per_seq_output_tokens

    def _score_candidate_sequence(self, seq_record: Dict[str, Any], candidate_sequence: str) -> float:
        """후보 시퀀스의 점수를 계산한다."""
        if not HOTPOTQA_AVAILABLE:
            return 0.0

        try:
            extra_info = {
                'supporting_facts': seq_record.get('supporting_payload', {}),
                'trainer_config': {
                    'give_partial_reward': True,
                    'use_answer_in_search_reward': self.config.get('use_answer_in_search_reward', False),
                    'require_search_match_for_answer': self.config.get('require_search_match_for_answer', False),
                    'partial_reward_weight': self.config.get('partial_reward_weight', 0.5),
                }
            }
            score = compute_score(
                solution_str=candidate_sequence,
                ground_truth=seq_record.get('ground_truth', ""),
                method='strict',
                format_score=0.5,
                score=1.0,
                extra_info=extra_info
            )
            return float(score)
        except Exception:
            return 0.0
    
    def _perform_batch_limited_search_resampling(self, groups_to_resample, current_n):
        """그룹 내 최고 시퀀스를 복제해 후보를 생성한 뒤 교체 전략을 계산한다."""

        truncated_input_ids = []
        remaining_searches = []
        sequence_records = []

        for group_idx, group_info in enumerate(groups_to_resample):
            best_entry = group_info.get('best_sequence', {})
            original_indices = group_info.get('original_indices', [])
            group_size = len(original_indices)
            if not best_entry or group_size <= 0:
                continue

            truncated_prompt = best_entry.get('truncated_prompt', "")
            prompt_token_ids = self.tokenizer.encode(truncated_prompt, add_special_tokens=False)
            remaining_budget = best_entry.get('remaining_searches', 0)
            if remaining_budget is None:
                remaining_budget = 0
            remaining_budget = int(remaining_budget)
            ground_truth = best_entry.get('ground_truth', group_info.get('default_ground_truth', ""))
            supporting_payload = best_entry.get('supporting_facts_payload', {})

            for seq_idx in original_indices:
                truncated_input_ids.append(prompt_token_ids.copy())
                remaining_searches.append(remaining_budget)
                sequence_records.append({
                    'group_idx': group_idx,
                    'orig_seq_idx': seq_idx,
                    'supporting_payload': supporting_payload if supporting_payload else {},
                    'ground_truth': ground_truth,
                    'original_score': float(best_entry.get('original_score', 0.0))
                })

        if not sequence_records:
            return {}, {'prompt_tokens': 0, 'response_tokens': 0}, {}

        prefix_list, response_list, token_stats, per_seq_output_tokens = self._run_batch_resample_generation(
            truncated_input_ids,
            remaining_searches,
        )

        group_candidates = {}
        resample_output_tokens_by_seq_idx = {}
        for record, prefix_str, response_str, output_tokens in zip(sequence_records, prefix_list, response_list, per_seq_output_tokens):
            candidate_sequence = prefix_str + response_str
            candidate_score = self._score_candidate_sequence(record, candidate_sequence)
            bucket = group_candidates.setdefault(record['group_idx'], [])
            bucket.append({
                'sequence': candidate_sequence,
                'score': float(candidate_score)
            })
            seq_idx = record.get('orig_seq_idx')
            if seq_idx is not None:
                resample_output_tokens_by_seq_idx[int(seq_idx)] = int(output_tokens)

        actions = {}
        for group_idx, group_info in enumerate(groups_to_resample):
            candidates = group_candidates.get(group_idx, [])
            candidate_scores = [entry['score'] for entry in candidates]
            candidate_sequences = [entry['sequence'] for entry in candidates]
            duplicate_count = len(candidate_sequences) - len(set(candidate_sequences)) if candidate_sequences else 0

            best_candidate = max(candidates, key=lambda item: item['score'], default=None)
            best_original_score = float(group_info.get('best_score', 0.0))
            worst_original_score = float(group_info.get('worst_score', 0.0))
            worst_sequence_idx = group_info.get('worst_sequence_idx')

            replacements = []
            if best_candidate and worst_sequence_idx is not None and best_candidate['score'] > best_original_score:
                replacements.append({
                    'target_index': worst_sequence_idx,
                    'target_score': worst_original_score,
                    'candidate_sequence': best_candidate['sequence'],
                    'candidate_score': float(best_candidate['score']),
                    'delta': float(best_candidate['score'] - worst_original_score)
                })

            prompt_id = group_info.get('prompt_id')
            candidates_round = [round(score, 3) for score in candidate_scores]
            selected_round = round(best_candidate['score'], 3) if best_candidate else None
            print(f"[OTR][Group {prompt_id}] best_orig={best_original_score:.3f} worst_orig={worst_original_score:.3f} candidates={candidates_round} selected={selected_round} attempts={len(candidates)} condition={bool(replacements)}")
            if duplicate_count:
                print(f"[OTR][DuplicateCheck] prompt={prompt_id} duplicates={duplicate_count}")

            plan_summary = {
                'prompt_id': prompt_id,
                'original_scores_sorted': [float(s) for s in group_info.get('original_scores', [])],
                'best_original_score': best_original_score,
                'worst_original_score': worst_original_score,
                'candidate_scores': candidate_scores,
                'selected_candidate_score': best_candidate['score'] if best_candidate else None,
                'duplicate_count': duplicate_count,
                'attempt_count': len(candidates),
                'condition_met': bool(replacements)
            }

            actions[group_idx] = {
                'group_info': group_info,
                'action': 'replace_many' if replacements else 'no_change',
                'replacements': replacements,
                'summary': plan_summary
            }

        return actions, token_stats, resample_output_tokens_by_seq_idx

    def _process_search_answer_batch_with_individual_limits(self, ans_list, current_prefix_list, sequences_reach_limit):
        """개별 시퀀스별 검색 제한을 적용한 배치 검색 처리"""
        search_queries = []
        ans_modified_list = []
        
        for ans in ans_list:
            search_query, ans_modified = check_search_token(ans.split("</think>")[0])
            search_queries.append(search_query)
            ans_modified_list.append(ans_modified)
        
        new_prefix_list = [None] * len(current_prefix_list)  
        search_flag_list = [False] * len(current_prefix_list)  
        
        # 각 시퀀스별로 개별 제한 적용
        for i, (search_query, reach_limit) in enumerate(zip(search_queries, sequences_reach_limit)):
            if search_query == "":
                new_prefix_list[i] = current_prefix_list[i]  
                search_flag_list[i] = False
            else:
                if reach_limit:
                    # 해당 시퀀스가 검색 제한에 도달한 경우
                    new_prefix = current_prefix_list[i] + f"{ans_modified_list[i]}\n\n<search_result> Reach the limit of search times. </search_result>\n\n"
                    new_prefix_list[i] = new_prefix  
                    search_flag_list[i] = True
                else:
                    # 아직 검색 가능한 경우
                    search_flag_list[i] = True
        
        # 실제 검색 수행 (제한에 도달하지 않은 시퀀스들만)
        if any(flag and not limit for flag, limit in zip(search_flag_list, sequences_reach_limit)):
            try:
                queries_to_search = []
                query_indices = []  
                
                for i, (flag, reach_limit) in enumerate(zip(search_flag_list, sequences_reach_limit)):
                    if flag and not reach_limit:  # 검색 필요하고 제한 안 도달
                        queries_to_search.append(search_queries[i])
                        query_indices.append(i)
                
                if queries_to_search:
                    search_results = get_search_results(queries_to_search)
                    
                    assert len(search_results) == len(queries_to_search), f"검색 결과 수와 쿼리 수가 일치하지 않습니다: {len(search_results)} vs {len(queries_to_search)}"
                    
                    for idx, result_idx in enumerate(query_indices):
                        if idx < len(search_results):  
                            new_prefix = current_prefix_list[result_idx] + f"{ans_modified_list[result_idx]}\n\n{search_results[idx]}\n\n"
                            new_prefix_list[result_idx] = new_prefix  
            
            except Exception as e:
                print("An error occurred during the batch search process: ", e)
                for i in range(len(search_queries)):
                    if search_flag_list[i] and new_prefix_list[i] is None:
                        new_prefix = current_prefix_list[i] + f"{ans_modified_list[i]}\n\n<search_result> Retrieval failed due to an error. </search_result>\n\n"
                        new_prefix_list[i] = new_prefix
        
        return new_prefix_list, search_flag_list
        
    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # rebuild vllm cache engine
        if _uses_legacy_vllm_api() and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()

        idx = prompts.batch['input_ids']  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch['attention_mask']
        position_ids = prompts.batch['position_ids']

        # used to construct attention_mask
        eos_token_id_raw = prompts.meta_info.get('eos_token_id')
        eos_token_id, eos_token_ids = _normalize_eos_token_ids(eos_token_id_raw)
        if eos_token_id is None:
            eos_token_id, eos_token_ids = _normalize_eos_token_ids(self.tokenizer.eos_token_id)
        if eos_token_id is None:
            raise ValueError("Unable to determine eos_token_id from prompts or tokenizer.")
        eos_token_id_set = set(eos_token_ids) if eos_token_ids else {eos_token_id}

        batch_size = idx.size(0)

        idx_list = []
        # parse idx from torch.Tensor to List[List[str]]
        for i in range(batch_size):
            idx_list.append(_pre_process_inputs(self.pad_token_id, idx[i]))

        do_sample = prompts.meta_info.get('do_sample', True)
        import time
        start_time = time.time()
        pre_gen_prompt_tokens_total = 0
        pre_gen_response_tokens_total = 0
        if not do_sample:
            kwargs = {
                'best_of': 1,
                'top_p': 1.0,
                'top_k': -1,
                'min_p': 0.0,
                'temperature': 0,
                'n': 1  # if greedy, only 1 response
            }

        # 🔧 검증 모드일 때 OTR sampling 비활성화
        is_validation = prompts.meta_info.get('validate', False)
        use_otr_for_this_generation = self.use_otr_sampling and not is_validation
        
        # Validation/Training mode detection

        # current_n을 미리 계산
        current_n = 1
        with self.update_sampling_params(**kwargs):
            current_n = self.sampling_params.n
            
            # OTR sampling을 위한 supporting_facts 및 ground_truth 추출 (current_n 사용)
            supporting_facts_list = []
            ground_truth_list = []
            if use_otr_for_this_generation:
                
                # supporting_facts 추출
                if 'supporting_facts' in prompts.non_tensor_batch:
                    supporting_facts_array = prompts.non_tensor_batch['supporting_facts']
                else:
                    supporting_facts_array = None
                        # supporting_facts key not found in non_tensor_batch
                
                # ground_truth 추출
                if 'ground_truth' in prompts.non_tensor_batch:
                    ground_truth_array = prompts.non_tensor_batch['ground_truth']
                    # Found ground_truth in prompts
                else:
                    ground_truth_array = None
                    # ground_truth key not found in non_tensor_batch
                
                # supporting_facts는 원본 프롬프트 기준이므로 실제 프롬프트 수 계산
                actual_prompt_count = batch_size  # 현재 워커가 받은 실제 프롬프트 수
                available_facts_count = len(supporting_facts_array) if supporting_facts_array is not None else 0
                # Batch size and supporting facts info available
                
                # 로컬 보관: 나중에 actual_n에 맞춰 재구성 가능하도록 저장
                local_supporting_facts_per_prompt = []
                local_ground_truth_per_prompt = []
                
                for i in range(batch_size):
                    # supporting_facts_array 범위 확인 및 안전한 접근
                    if supporting_facts_array is not None and i < available_facts_count:
                        supporting_facts = supporting_facts_array[i]
                    else:
                        supporting_facts = {}
                    
                    # 항상 로컬 인덱스를 original_prompt_id로 사용 (global id 무시)
                    original_prompt_id = i
                    
                    # ground_truth 추출
                    if ground_truth_array is not None and i < len(ground_truth_array):
                        ground_truth = ground_truth_array[i]
                    else:
                        ground_truth = ""
                    
                    local_supporting_facts_per_prompt.append({
                        'original_prompt_id': original_prompt_id,
                        'supporting_facts': supporting_facts
                    })
                    local_ground_truth_per_prompt.append(ground_truth)
                    
                    # 각 프롬프트마다 current_n개의 시퀀스 생성
                    for seq_idx_in_prompt in range(current_n):
                        facts_with_meta = {
                            'original_prompt_id': original_prompt_id,
                            'seq_idx_in_prompt': seq_idx_in_prompt,
                            'supporting_facts': supporting_facts
                        }
                        supporting_facts_list.append(facts_with_meta)
                        ground_truth_list.append(ground_truth)
            
            else:
                # OTR sampling이 아닐 때는 빈 리스트로 초기화
                supporting_facts_list = [{}] * (batch_size * current_n)
                ground_truth_list = [""] * (batch_size * current_n)
            
            tmp_sampling_params = copy.deepcopy(self.sampling_params)
            tmp_sampling_params.max_tokens = 2048
            tmp_sampling_params.stop_token_ids = list(eos_token_ids) if eos_token_ids else [eos_token_id]
            output = self.inference_engine.generate(
                prompts=None,  # because we have already convert it to prompt token id
                sampling_params=tmp_sampling_params,
                prompt_token_ids=idx_list,
                use_tqdm=False)

        response = _extract_output_token_ids(output)
        try:
            pre_output_tokens_per_seq = [len(ids) for ids in response]
        except Exception as _pre_e:
            print(f"[OTR] pre-output token init failed: {type(_pre_e).__name__}: {_pre_e}")
            pre_output_tokens_per_seq = [0] * len(response)
        try:
            if idx_list:
                prompt_tokens = sum(len(ids) for ids in idx_list)
                sequences_per_prompt = max(1, len(response) // len(idx_list))
                pre_gen_prompt_tokens_total += prompt_tokens * sequences_per_prompt
            pre_gen_response_tokens_total += sum(len(ids) for ids in response)
        except Exception as _pre_e:
            print(f"[OTR] pre-token count (initial) failed: {type(_pre_e).__name__}: {_pre_e}")
        
        current_prefix_list = []
        for sample_idx in range(len(idx_list)):
            for _n in range(current_n):
                current_prefix_list.append(idx_list[sample_idx])
        
        current_prefix_list = self.tokenizer.batch_decode(current_prefix_list,skip_special_tokens=False)
        response_str_list = self.tokenizer.batch_decode(response,skip_special_tokens=False)
        
        # 출력 실제 개수 기준으로 actual_n 재계산 및 메타데이터 동기화
        try:
            total_samples = len(response_str_list)
            base_prompts = len(idx_list)
            if base_prompts > 0 and total_samples % base_prompts == 0:
                actual_n = total_samples // base_prompts
            else:
                actual_n = current_n
            
            if actual_n != current_n:
                # 프리픽스 재구성
                rebuilt_prefix_token_ids = []
                for sample_idx in range(len(idx_list)):
                    for _n in range(actual_n):
                        rebuilt_prefix_token_ids.append(idx_list[sample_idx])
                current_prefix_list = self.tokenizer.batch_decode(rebuilt_prefix_token_ids, skip_special_tokens=False)
                
                # OTR 메타데이터 재빌드
                if use_otr_for_this_generation:
                    new_supporting_facts_list = []
                    new_ground_truth_list = []
                    for i in range(base_prompts):
                        # 로컬 백업이 있는 경우 사용, 없으면 최소 정보로 대체
                        if 'local_supporting_facts_per_prompt' in locals() and i < len(local_supporting_facts_per_prompt):
                            sf_meta = local_supporting_facts_per_prompt[i]
                            sf = sf_meta.get('supporting_facts', {})
                        else:
                            sf = {}
                        gt = local_ground_truth_per_prompt[i] if 'local_ground_truth_per_prompt' in locals() and i < len(local_ground_truth_per_prompt) else ""
                        for seq_idx_in_prompt in range(actual_n):
                            new_supporting_facts_list.append({
                                'original_prompt_id': i,
                                'seq_idx_in_prompt': seq_idx_in_prompt,
                                'supporting_facts': sf
                            })
                            new_ground_truth_list.append(gt)
                    supporting_facts_list = new_supporting_facts_list
                    ground_truth_list = new_ground_truth_list
                else:
                    supporting_facts_list = [{}] * (base_prompts * actual_n)
                    ground_truth_list = [""] * (base_prompts * actual_n)
                
                current_n = actual_n
        except Exception as _e:
            # 동기화 실패 시 기존 로직 유지 (안전장치)
            pass
        
        raw_current_prefix_list = copy.deepcopy(current_prefix_list)
        
       
        re_sampling_params = copy.deepcopy(self.sampling_params)
        re_sampling_params.n = 1
        re_sampling_params.max_tokens=1024
        re_sampling_params.stop_token_ids=list(eos_token_ids) if eos_token_ids else [eos_token_id]

        print('re_sampling_params', re_sampling_params)
        pber = tqdm(range(self.max_search_nums + 1), desc="Searching...", disable=False)        
        
        for iter in pber:
            assert len(response_str_list) == len(current_prefix_list), \
                f"response_str_list and current_prefix_list should have the same length, but got {len(response_str_list)} and {len(current_prefix_list)}"
            
            if iter == self.max_search_nums:
                re_sampling_params.max_tokens = 2048
                
            start_time = time.time()
            new_prefix_list, search_flag_list = process_search_answer_batch(
                response_str_list, 
                current_prefix_list, 
                reach_limit=(iter == self.max_search_nums)
            )
            
            current_prefix_list = new_prefix_list
            
            pber.set_description(f"Done for Searching...{iter+1}, cost time: {time.time() - start_time:.2f}s, now re-generating {sum(search_flag_list)}/{len(search_flag_list)} samples")
            
            if any(search_flag_list):
                # 검색이 필요한 시퀀스들만 수집
                new_prompts_list = [current_prefix for search_flag, current_prefix in zip(search_flag_list, current_prefix_list) if search_flag]
                
                if new_prompts_list:
                    input_ids_list = [self.tokenizer.encode(p, add_special_tokens=False) for p in new_prompts_list]
                    
                    output = self.inference_engine.generate(
                        prompts=None,
                        sampling_params=re_sampling_params,
                        prompt_token_ids=input_ids_list,
                        use_tqdm=False
                    )
                    
                    new_response_list = _extract_output_token_ids(output)
                    try:
                        pre_gen_prompt_tokens_total += sum(len(ids) for ids in input_ids_list)
                        pre_gen_response_tokens_total += sum(len(ids) for ids in new_response_list)
                    except Exception as _pre_e:
                        print(f"[OTR] pre-token count (regen) failed: {type(_pre_e).__name__}: {_pre_e}")
                    try:
                        new_response_lengths = [len(ids) for ids in new_response_list]
                    except Exception as _pre_e:
                        print(f"[OTR] pre-output token regen failed: {type(_pre_e).__name__}: {_pre_e}")
                        new_response_lengths = []
                    new_response_str_list = self.tokenizer.batch_decode(new_response_list, skip_special_tokens=False)
                    
                    # 응답 업데이트
                    updated_response_str_list = []
                    new_response_idx = 0
                    
                    for i, search_flag in enumerate(search_flag_list):
                        if search_flag and new_response_idx < len(new_response_str_list):
                            updated_response_str_list.append(new_response_str_list[new_response_idx])
                            if i < len(pre_output_tokens_per_seq) and new_response_idx < len(new_response_lengths):
                                pre_output_tokens_per_seq[i] += new_response_lengths[new_response_idx]
                            new_response_idx += 1
                        else:
                            # 기존 응답 유지
                            updated_response_str_list.append(response_str_list[i] if i < len(response_str_list) else "")
                    
                    response_str_list = updated_response_str_list
            else:
                break
            
        pber.close()
        
        # 일반 생성 완료
        try:
            pre_prompt_tokens = [int(pre_gen_prompt_tokens_total)]
            pre_response_tokens = [int(pre_gen_response_tokens_total)]
        except Exception as _len_e:
            print(f"[OTR] pre-token count failed: {type(_len_e).__name__}: {_len_e}")
            pre_prompt_tokens = [0]
            pre_response_tokens = [0]

        otr_resample_prompt_tokens_total = 0
        otr_resample_response_tokens_total = 0
        otr_cut_offsets = [0] * len(response_str_list)
        otr_cut_output_tokens = [0] * len(response_str_list)
        otr_output_tokens_per_seq = [0] * len(response_str_list)
        otr_group_attempted_flags = [False] * len(response_str_list)
        
        # 🔒 정렬 보장: 일반 생성 결과를 기반으로 per-sequence 메타를 확정 수집
        try:
            total_sequences = len(response_str_list)
            base_prompts = len(idx_list)
            assert total_sequences == len(current_prefix_list), "response/prefix length mismatch"
            assert base_prompts > 0, "base_prompts must be > 0"
            sequences_per_prompt = max(1, total_sequences // base_prompts)
            
            assembled_full_sequences = []
            assembled_supporting_facts_list = []
            assembled_ground_truths = []
            assembled_prompt_lengths = []
            
            for i in range(total_sequences):
                original_prompt_idx = i // sequences_per_prompt
                if original_prompt_idx >= base_prompts:
                    original_prompt_idx = base_prompts - 1
                
                # 1) 전체 시퀀스 저장
                assembled_full_sequences.append(current_prefix_list[i] + response_str_list[i])
                
                # 2) per-prompt supporting_facts 및 ground_truth 정렬 수집
                if use_otr_for_this_generation and 'local_supporting_facts_per_prompt' in locals():
                    if original_prompt_idx < len(local_supporting_facts_per_prompt):
                        sf_meta = local_supporting_facts_per_prompt[original_prompt_idx]
                        if isinstance(sf_meta, dict):
                            sf_payload = sf_meta.get('supporting_facts', {})
                        else:
                            sf_payload = sf_meta
                    else:
                        sf_payload = {}
                    assembled_supporting_facts_list.append({
                        'original_prompt_id': original_prompt_idx,
                        'supporting_facts': sf_payload
                    })
                    if 'local_ground_truth_per_prompt' in locals() and original_prompt_idx < len(local_ground_truth_per_prompt):
                        gt_val = local_ground_truth_per_prompt[original_prompt_idx]
                    else:
                        gt_val = ""
                    assembled_ground_truths.append(gt_val)
                else:
                    assembled_supporting_facts_list.append({})
                    assembled_ground_truths.append("")
                
                # 3) 프롬프트 길이 (원본 프롬프트 텍스트 기준)
                if original_prompt_idx < len(raw_current_prefix_list):
                    pl = len(raw_current_prefix_list[original_prompt_idx])
                else:
                    pl = 0
                assembled_prompt_lengths.append(pl)
        except Exception as _align_e:
            print(f"[OTR] Assembly alignment failed: {type(_align_e).__name__}: {_align_e}")
        
        # 새로운 OTR 로직 적용 (use_otr_sampling이 True일 때만)
        if use_otr_for_this_generation and self.use_otr_sampling:  # 🚫 OTR 임시 비활성화
            # 🔧 supporting_facts_list와 response_str_list 길이 확인
            if len(supporting_facts_list) != len(response_str_list):
                error_msg = f"CRITICAL ERROR: Length mismatch detected!\n"
                error_msg += f"supporting_facts_list: {len(supporting_facts_list)}\n"
                error_msg += f"response_str_list: {len(response_str_list)}\n"
                error_msg += f"current_n: {current_n}\n"
                error_msg += f"This indicates a bug in OTR metadata synchronization!"
                
                print(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            # 🎯 전체 시퀀스 생성
            full_sequences = []
            for i, (response_str, current_prefix) in enumerate(zip(response_str_list, current_prefix_list)):
                full_sequence = current_prefix + response_str
                full_sequences.append(full_sequence)
            
            # ground_truth 리스트 조정
            current_ground_truths = []
            for i in range(len(full_sequences)):
                if i < len(ground_truth_list):
                    current_ground_truths.append(ground_truth_list[i])
                else:
                    # supporting_facts_list에서 원본 프롬프트 ID를 찾아서 해당 ground_truth 사용
                    if i < len(supporting_facts_list):
                        facts_meta = supporting_facts_list[i]
                        if isinstance(facts_meta, dict) and 'original_prompt_id' in facts_meta:
                            original_id = facts_meta['original_prompt_id']
                            if original_id < len(ground_truth_list):
                                current_ground_truths.append(ground_truth_list[original_id])
                            else:
                                current_ground_truths.append("")
                        else:
                            current_ground_truths.append("")
                    else:
                        current_ground_truths.append("")
            
            # ✅ 확정 수집한 정렬 데이터를 사용
            if 'assembled_full_sequences' in locals():
                full_sequences = assembled_full_sequences
            if 'assembled_supporting_facts_list' in locals():
                supporting_facts_list = assembled_supporting_facts_list
            if 'assembled_ground_truths' in locals():
                current_ground_truths = assembled_ground_truths
            
            # 🆕 새로운 OTR 재샘플링 로직 적용 (프롬프트 길이 정보 포함)
            prompt_lengths = []
            for i, facts_meta in enumerate(supporting_facts_list):
                if isinstance(facts_meta, dict) and 'original_prompt_id' in facts_meta:
                    original_id = facts_meta['original_prompt_id']
                    if original_id >= len(raw_current_prefix_list):
                        raise ValueError(f"Invalid original_prompt_id {original_id} at sequence {i}, raw_current_prefix_list length: {len(raw_current_prefix_list)}")
                    prompt_lengths.append(len(raw_current_prefix_list[original_id]))
                else:
                    raise ValueError(f"Missing original_prompt_id in supporting_facts_list at index {i}: {facts_meta}")
            
            groups_to_resample, final_sequences, final_supporting_facts = new_otr_resampling_logic(
                full_sequences=full_sequences,
                supporting_facts_list=supporting_facts_list,
                ground_truths=current_ground_truths,
                prompt_lengths=prompt_lengths,
                score_threshold=1.0,
                rollout_n=current_n,
                max_search_nums=self.max_search_nums
            )

            prompt_token_lens = {}
            try:
                base_prompts = len(idx_list)
                for i in range(base_prompts):
                    prompt_token_lens[i] = len(self.tokenizer.encode(raw_current_prefix_list[i], add_special_tokens=False))
            except Exception as _tok_e:
                print(f"[OTR] prompt token length failed: {type(_tok_e).__name__}: {_tok_e}")
                prompt_token_lens = {}

            if groups_to_resample:
                for group_info in groups_to_resample:
                    cut_offset = int(group_info.get('cut_offset', 0) or 0)
                    prompt_id = group_info.get('prompt_id')
                    prompt_token_len = int(prompt_token_lens.get(prompt_id, 0))
                    truncated_prompt = ""
                    best_entry = group_info.get('best_sequence', {})
                    if isinstance(best_entry, dict):
                        truncated_prompt = best_entry.get('truncated_prompt', "") or ""
                    try:
                        truncated_token_len = len(self.tokenizer.encode(truncated_prompt, add_special_tokens=False))
                    except Exception:
                        truncated_token_len = 0
                    cut_output_tokens = max(0, truncated_token_len - prompt_token_len)
                    for seq_idx in group_info.get('original_indices', []):
                        if seq_idx < len(otr_cut_offsets):
                            otr_cut_offsets[seq_idx] = cut_offset
                            otr_cut_output_tokens[seq_idx] = cut_output_tokens
                            otr_group_attempted_flags[seq_idx] = True
            
            # 재샘플링 수행 (배치 처리)
            otr_replacement_plan_map = {}
            if groups_to_resample:
                # Performing batch resampling
                
                try:
                    # 🔧 새로운 배치 재샘플링 수행 (치환 액션 계산)
                    actions, resample_token_stats, resample_output_tokens_by_seq_idx = self._perform_batch_limited_search_resampling(
                        groups_to_resample,
                        current_n
                    )
                    otr_resample_prompt_tokens_total = int(resample_token_stats.get('prompt_tokens', 0))
                    otr_resample_response_tokens_total = int(resample_token_stats.get('response_tokens', 0))
                    for seq_idx, tok_count in resample_output_tokens_by_seq_idx.items():
                        if seq_idx < len(otr_output_tokens_per_seq):
                            otr_output_tokens_per_seq[seq_idx] = int(tok_count)
                    
                    # 최종 시퀀스에 치환 적용 (추가 금지)
                    replaced_flags_per_sequence = [False] * len(final_sequences)
                    resampled_group_flags = {}  # prompt_id -> bool
                    otr_replacement_plan_map = {}
                    
                    for group_idx, act in actions.items():
                        group_info = act['group_info']
                        original_indices = group_info['original_indices']
                        prompt_id = group_info['prompt_id']
                        
                        plan_summary = act.get('summary')
                        if plan_summary:
                            otr_replacement_plan_map[str(prompt_id)] = copy.deepcopy(plan_summary)
                        
                        action_type = act.get('action')
                        if action_type == 'replace_all':
                            resampled_group_flags[prompt_id] = True
                            new_full_sequences = act['new_sequences']
                            for j, seq_idx in enumerate(original_indices):
                                if j < len(new_full_sequences) and seq_idx < len(final_sequences):
                                    final_sequences[seq_idx] = new_full_sequences[j]
                                    replaced_flags_per_sequence[seq_idx] = True
                        elif action_type == 'replace_one':
                            resampled_group_flags[prompt_id] = True
                            target_idx = group_info.get('worst_sequence_idx', group_info['best_sequence_idx'])
                            if target_idx < len(final_sequences):
                                final_sequences[target_idx] = act['new_sequence']
                                replaced_flags_per_sequence[target_idx] = True
                        elif action_type == 'replace_many':
                            replacements = act.get('replacements', [])
                            if replacements:
                                resampled_group_flags[prompt_id] = True
                                for repl in replacements:
                                    target_idx = repl['target_index']
                                    if target_idx < len(final_sequences):
                                        final_sequences[target_idx] = repl['candidate_sequence']
                                        replaced_flags_per_sequence[target_idx] = True
                            else:
                                resampled_group_flags.setdefault(prompt_id, False)
                        else:
                            resampled_group_flags.setdefault(prompt_id, False)
                    
                    # 재구성 이후 metadata를 위해 저장
                    otr_replaced_flags = replaced_flags_per_sequence
                    otr_group_resampled_map = resampled_group_flags
                    
                except Exception as e:
                    # Batch resampling failed, falling back to original sequences
                    otr_replaced_flags = [False] * len(final_sequences)
                    otr_group_resampled_map = {}
                    otr_replacement_plan_map = {}
                    otr_resample_prompt_tokens_total = 0
                    otr_resample_response_tokens_total = 0
                    otr_output_tokens_per_seq = [0] * len(final_sequences)
            else:
                otr_replaced_flags = [False] * len(final_sequences)
                otr_group_resampled_map = {}
                otr_replacement_plan_map = {}
                otr_resample_prompt_tokens_total = 0
                otr_resample_response_tokens_total = 0
                otr_output_tokens_per_seq = [0] * len(final_sequences)
            
            # 최종 결과를 response_str_list로 변환
            response_str_list = []
            current_prefix_list = []
            supporting_facts_list = final_supporting_facts
            
            for i, full_seq in enumerate(final_sequences):
                # 원본 프롬프트 길이 계산
                if i < len(supporting_facts_list):
                    facts_meta = supporting_facts_list[i]
                    if isinstance(facts_meta, dict) and 'original_prompt_id' in facts_meta:
                        original_id = facts_meta['original_prompt_id']
                        if original_id < len(raw_current_prefix_list):
                            prefix_len = len(raw_current_prefix_list[original_id])
                        else:
                            prefix_len = len(raw_current_prefix_list[0]) if raw_current_prefix_list else 0
                    else:
                        prefix_len = len(raw_current_prefix_list[0]) if raw_current_prefix_list else 0
                else:
                    prefix_len = len(raw_current_prefix_list[0]) if raw_current_prefix_list else 0
                
                current_prefix = full_seq[:prefix_len]
                response_str = full_seq[prefix_len:]
                
                current_prefix_list.append(current_prefix)
                response_str_list.append(response_str)
            
            # New OTR completed
            
            full_content = []
            for response_str, current_prefix in zip(response_str_list, current_prefix_list):
                full_content.append(current_prefix + response_str)
            response = []
            
            for p_id, p in enumerate(full_content):
                # 그룹 기반 프리픽스 길이 계산 (fallback 경로)
                original_prompt_id = p_id // current_n if 'current_n' in locals() and current_n > 0 else 0
                if original_prompt_id < len(raw_current_prefix_list):
                    prefix_len = len(raw_current_prefix_list[original_prompt_id])
                else:
                    prefix_len = len(raw_current_prefix_list[0]) if raw_current_prefix_list else 0
                response.append(self.tokenizer.encode(p[prefix_len:], add_special_tokens=False))
            
            max_response_len = -1
            
            
            for i_res in range(len(response)):
                while len(response[i_res]) > 1 and response[i_res][-1] in eos_token_id_set:
                    if response[i_res][-2] in eos_token_id_set:
                        response[i_res] = response[i_res][:-1]
                    else:
                        break
                
                
                max_response_len = max(max_response_len, len(response[i_res]))
                
                # response[i_res] = torch.tensor(response[i_res], device=attention_mask.device, dtype=attention_mask.dtype)
            
            # pad right to largest response
                    
            with self.update_sampling_params(**kwargs):
                        # max_tokens
                # print(f"max_response_len: {max_response_len}")
                # print(f"sampling_params.max_tokens: {self.sampling_params.max_tokens}")
                for i_res in range(len(response)):
                    response[i_res] = response[i_res] + [eos_token_id] * (max_response_len - len(response[i_res]))
                
                    response[i_res] = response[i_res][:self.sampling_params.max_tokens]
                    
            
            response = torch.tensor(response, device=attention_mask.device, dtype=attention_mask.dtype)
            
            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)
            # response = output[0].to(idx.device)
            # log_probs = output[1].to(idx.device)
            
            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                # log_probs = pad_sequence_to_length(log_probs, self.config.response_length, self.pad_token_id)
            
            
            # 배치 크기 조정 및 입력/마스크 병합 (OTR 경로에서 실행됨)
            actual_batch_size = len(response)
            original_batch_size = idx.size(0)
            if actual_batch_size != original_batch_size:
                sequences_per_prompt = actual_batch_size // original_batch_size
                idx = idx.repeat_interleave(sequences_per_prompt, dim=0)
                attention_mask = attention_mask.repeat_interleave(sequences_per_prompt, dim=0)
                position_ids = position_ids.repeat_interleave(sequences_per_prompt, dim=0)
                batch_size = actual_batch_size
            else:
                batch_size = original_batch_size

            seq = torch.cat([idx, response], dim=-1)

            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).repeat(batch_size, 1)

            # TODO(sgm): fix position_ids on right_pad
            # prompt: left pad + response: right pad
            # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
            # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
            response_position_ids = position_ids[:, -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            response_attention_mask = get_eos_mask(
                response_id=response,
                eos_token=eos_token_ids if eos_token_ids else eos_token_id,
                dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            # all the tp ranks should contain the same data here. data in all ranks are valid
            batch = TensorDict(
                {
                    'prompts': idx,
                    'responses': response,
                    'input_ids': seq,  # here input_ids become the whole sentences
                    # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                },
                    batch_size=batch_size)

        if 'otr_group_resampled_map' not in locals():
            otr_group_resampled_map = {}
        if 'otr_replaced_flags' not in locals():
            otr_replaced_flags = [False] * len(response)

        # 🔖 OTR 디버깅용 non-tensor 플래그 구성 (시퀀스 길이와 정렬)
        otr_group_ids = []
        otr_group_resampled_flags = []
        otr_replacement_summary_seq = []
        for i, facts_meta in enumerate(supporting_facts_list):
            if isinstance(facts_meta, dict) and 'original_prompt_id' in facts_meta:
                gid = facts_meta['original_prompt_id']
            else:
                gid = 0
            otr_group_ids.append(gid)
            # group index는 groups_to_resample의 순번이 아니라 prompt_id 기반이므로 매핑
            # resampled 여부는 prompt_id가 key인 맵에서 조회
            otr_group_resampled_flags.append(bool(otr_group_resampled_map.get(gid, False)))
            summary_for_gid = otr_replacement_plan_map.get(str(gid), {}) if 'otr_replacement_plan_map' in locals() else {}
            otr_replacement_summary_seq.append(copy.deepcopy(summary_for_gid))
        
        # Fallback: Non-OTR 경로에서 배치 미구성 시 공통 배치 빌드
        if 'batch' not in locals():
            full_content = []
            for response_str, current_prefix in zip(response_str_list, current_prefix_list):
                full_content.append(current_prefix + response_str)
            response = []
            
            for p_id, p in enumerate(full_content):
                # 그룹 기반 프리픽스 길이 계산 (fallback 경로)
                original_prompt_id = p_id // current_n if 'current_n' in locals() and current_n > 0 else 0
                if original_prompt_id < len(raw_current_prefix_list):
                    prefix_len = len(raw_current_prefix_list[original_prompt_id])
                else:
                    prefix_len = len(raw_current_prefix_list[0]) if raw_current_prefix_list else 0
                response.append(self.tokenizer.encode(p[prefix_len:], add_special_tokens=False))
            
            max_response_len = -1
            for i_res in range(len(response)):
                while len(response[i_res]) > 1 and response[i_res][-1] in eos_token_id_set:
                    if response[i_res][-2] in eos_token_id_set:
                        response[i_res] = response[i_res][:-1]
                    else:
                        break
                max_response_len = max(max_response_len, len(response[i_res]))
            
            with self.update_sampling_params(**kwargs):
                for i_res in range(len(response)):
                    response[i_res] = response[i_res] + [eos_token_id] * (max_response_len - len(response[i_res]))
                    response[i_res] = response[i_res][:self.sampling_params.max_tokens]
            
            response = torch.tensor(response, device=attention_mask.device, dtype=attention_mask.dtype)
            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            
            # 배치 크기 조정 (fallback에서도 동일 로직 적용)
            actual_batch_size = len(response)
            original_batch_size = idx.size(0)
            if actual_batch_size != original_batch_size:
                sequences_per_prompt = actual_batch_size // original_batch_size
                idx = idx.repeat_interleave(sequences_per_prompt, dim=0)
                attention_mask = attention_mask.repeat_interleave(sequences_per_prompt, dim=0)
                position_ids = position_ids.repeat_interleave(sequences_per_prompt, dim=0)
                batch_size = actual_batch_size
            else:
                batch_size = original_batch_size
            seq = torch.cat([idx, response], dim=-1)
            response_length = response.size(1)
            delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
            response_position_ids = position_ids[:, -1:] + delta_position_id
            position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
            response_attention_mask = get_eos_mask(
                response_id=response,
                eos_token=eos_token_ids if eos_token_ids else eos_token_id,
                dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)
            batch = TensorDict(
                {
                    'prompts': idx,
                    'responses': response,
                    'input_ids': seq,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                },
                batch_size=batch_size)
        
        if 'otr_replacement_plan_map' not in locals():
            otr_replacement_plan_map = {}
        if 'otr_replacement_summary_seq' not in locals():
            otr_replacement_summary_seq = [{}]*batch_size

        # free vllm cache engine
        if _uses_legacy_vllm_api() and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()
        
        def _align_list(vals, target_len, fill):
            if vals is None:
                return [fill] * target_len
            if len(vals) < target_len:
                return vals + [fill] * (target_len - len(vals))
            return vals[:target_len]

        pre_prompt_tokens = _align_list(pre_prompt_tokens if 'pre_prompt_tokens' in locals() else [], batch_size, 0)
        pre_response_tokens = _align_list(pre_response_tokens if 'pre_response_tokens' in locals() else [], batch_size, 0)
        pre_output_tokens_per_seq = _align_list(pre_output_tokens_per_seq if 'pre_output_tokens_per_seq' in locals() else [], batch_size, 0)
        otr_cut_offsets = _align_list(otr_cut_offsets if 'otr_cut_offsets' in locals() else [], batch_size, 0)
        otr_cut_output_tokens = _align_list(otr_cut_output_tokens if 'otr_cut_output_tokens' in locals() else [], batch_size, 0)
        otr_output_tokens_per_seq = _align_list(otr_output_tokens_per_seq if 'otr_output_tokens_per_seq' in locals() else [], batch_size, 0)
        otr_group_attempted_flags = _align_list(otr_group_attempted_flags if 'otr_group_attempted_flags' in locals() else [], batch_size, False)

        otr_resample_prompt_tokens = [0] * batch_size
        otr_resample_response_tokens = [0] * batch_size
        if batch_size > 0:
            otr_resample_prompt_tokens[0] = int(otr_resample_prompt_tokens_total) if 'otr_resample_prompt_tokens_total' in locals() else 0
            otr_resample_response_tokens[0] = int(otr_resample_response_tokens_total) if 'otr_resample_response_tokens_total' in locals() else 0

        # 🔖 OTR 디버깅 정보를 non_tensor_batch에 포함
        result_proto = DataProto(batch=batch)
        try:
            import numpy as _np
            result_proto.non_tensor_batch = {
                'otr_replaced_flags': _np.array(otr_replaced_flags if 'otr_replaced_flags' in locals() else [False]*batch_size, dtype=object),
                'otr_group_ids': _np.array(otr_group_ids if 'otr_group_ids' in locals() else [0]*batch_size, dtype=object),
                'otr_group_resampled_flags': _np.array(otr_group_resampled_flags if 'otr_group_resampled_flags' in locals() else [False]*batch_size, dtype=object),
                'otr_replacement_summary': _np.array(otr_replacement_summary_seq if 'otr_replacement_summary_seq' in locals() else [{}]*batch_size, dtype=object),
                'otr_pre_prompt_tokens': _np.array(pre_prompt_tokens, dtype=object),
                'otr_pre_response_tokens': _np.array(pre_response_tokens, dtype=object),
                'pre_output_tokens_per_seq': _np.array(pre_output_tokens_per_seq, dtype=object),
                'otr_resample_prompt_tokens': _np.array(otr_resample_prompt_tokens, dtype=object),
                'otr_resample_response_tokens': _np.array(otr_resample_response_tokens, dtype=object),
                'otr_cut_offsets': _np.array(otr_cut_offsets, dtype=object),
                'otr_cut_output_tokens': _np.array(otr_cut_output_tokens, dtype=object),
                'otr_output_tokens_per_seq': _np.array(otr_output_tokens_per_seq, dtype=object),
                'otr_group_attempted_flags': _np.array(otr_group_attempted_flags, dtype=object),
            }
        except Exception:
            result_proto.non_tensor_batch = {}
        
        return result_proto
