#!/usr/bin/env python3
"""CPU-only regressions for the canonical OTR prompt boundary."""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]
ROLLOUT_PATH = ROOT / "verl/workers/rollout/vllm_rollout/vllm_rollout_coa.py"
REWARD_PATH = ROOT / "verl/utils/reward_score/hotpotqa.py"
MAPPING_PATH = ROOT / "verl/workers/rollout/vllm_rollout/otr_prefix_mapping.py"
MAPPING_SPEC = importlib.util.spec_from_file_location("otr_prefix_mapping", MAPPING_PATH)
assert MAPPING_SPEC is not None and MAPPING_SPEC.loader is not None
MAPPING = importlib.util.module_from_spec(MAPPING_SPEC)
MAPPING_SPEC.loader.exec_module(MAPPING)
canonical_prefixes_from_prompt_major_expansion = (
    MAPPING.canonical_prefixes_from_prompt_major_expansion
)
split_final_sequences_at_canonical_prefix = (
    MAPPING.split_final_sequences_at_canonical_prefix
)
LOCKED_FUNCTION_SHA256 = {
    "_perform_batch_limited_search_resampling": "a7ca337d9372ec0d636c90f2eb801ab5d24c13a8a1d471895e8b296d3fb1acb2",
    "_process_search_answer_batch_with_individual_limits": "7c5ee68cc328e8a7e8b9ceef455d704f0534d6772482211bc02a91dd24aa9155",
    "_run_batch_resample_generation": "b4ba061b3d0b583202a92d5a2d62b78df6a4a4dfd9f6fe358bc57676cd427666",
    "_score_candidate_sequence": "5e7bfbb3d2712e73176442eaa54d4e933dbc3479cd280caeb8d496f77b32c43f",
    "find_optimal_truncation_point": "75e292199c435efe4ffb38dcc388ab3a7dce0b91ba5d8a8da65d18a70444c215",
    "new_otr_resampling_logic": "1fd5d6704574d57a6cbddaabbcb6f408be1ff4d528d684a0a607903fa1c83303",
    "score_sequences_simple": "78b8c4b71160ed8c8376d242c553e7a9ebdecea72cc8c80dbdfd11d4f944f88a",
}
LOCKED_REWARD_SHA256 = "f1e784e08148c7726ecebfc61eef73ba667698a1915870118ed037a7fa8b5a1b"


def fixture(current_n: int = 5):
    canonical = [
        "<p0>",
        "<prompt-one>",
        "<prompt-two-is-longer>",
        "<prompt-three-has-a-different-size>",
        "<prompt-four-is-longer-than-every-prefix-before-it>",
        "<p5-is-a-distinct-final-prompt>",
    ]
    repeated = [prompt for prompt in canonical for _ in range(current_n)]
    metadata = []
    responses = []
    full_sequences = []
    for prompt_id, prompt in enumerate(canonical):
        for seq_idx in range(current_n):
            response = f"<answer prompt={prompt_id} sample={seq_idx}>"
            metadata.append(
                {"original_prompt_id": prompt_id, "seq_idx_in_prompt": seq_idx}
            )
            responses.append(response)
            full_sequences.append(prompt + response)
    return canonical, repeated, metadata, responses, full_sequences


def expect_failure(callable_obj) -> None:
    try:
        callable_obj()
    except (TypeError, ValueError):
        return
    raise AssertionError("malformed prefix layout did not fail closed")


def locked_function_hashes() -> dict[str, str]:
    source = ROLLOUT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    observed = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in LOCKED_FUNCTION_SHA256:
                segment = ast.get_source_segment(source, node)
                observed[node.name] = hashlib.sha256(segment.encode("utf-8")).hexdigest()
    return observed


def main() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", (
        "test must run with CUDA_VISIBLE_DEVICES=''"
    )

    canonical, repeated, metadata, expected, full_sequences = fixture()
    before = copy.deepcopy((canonical, repeated, metadata, expected, full_sequences))
    collapsed = canonical_prefixes_from_prompt_major_expansion(
        repeated, base_prompt_count=6, current_n=5
    )
    prefixes, responses, prompt_ids = split_final_sequences_at_canonical_prefix(
        full_sequences, metadata, collapsed, current_n=5
    )
    assert collapsed == canonical
    assert responses == expected
    assert prefixes == [canonical[prompt_id] for prompt_id in prompt_ids]
    assert prompt_ids == [prompt_id for prompt_id in range(6) for _ in range(5)]
    assert before == (canonical, repeated, metadata, expected, full_sequences)

    legacy_responses = []
    for sequence_index, full_sequence in enumerate(full_sequences):
        prompt_id = sequence_index // 5
        legacy_responses.append(full_sequence[len(repeated[prompt_id]) :])
    legacy_diff_count = sum(
        legacy != wanted for legacy, wanted in zip(legacy_responses, expected)
    )
    assert legacy_diff_count == 25
    duplicated_suffix_count = sum(
        actual != wanted and actual.endswith(wanted)
        for actual, wanted in zip(responses, expected)
    )
    dropped_head_count = sum(
        actual != wanted and wanted.endswith(actual)
        for actual, wanted in zip(responses, expected)
    )
    assert duplicated_suffix_count == 0
    assert dropped_head_count == 0

    replaced_flags = [False] * len(full_sequences)
    replaced_index = 8
    replaced_flags[replaced_index] = True
    replaced_sequences = list(full_sequences)
    replaced_sequences[replaced_index] = canonical[1] + "<resampled-better-answer>"
    _, replaced_responses, replaced_prompt_ids = split_final_sequences_at_canonical_prefix(
        replaced_sequences, metadata, collapsed, current_n=5
    )
    assert replaced_responses[replaced_index] == "<resampled-better-answer>"
    assert replaced_prompt_ids == prompt_ids
    assert all(
        replaced_responses[index] == expected[index]
        for index, was_replaced in enumerate(replaced_flags)
        if not was_replaced
    )

    n1_canonical, n1_repeated, n1_metadata, n1_expected, n1_full = fixture(1)
    n1_collapsed = canonical_prefixes_from_prompt_major_expansion(
        n1_repeated, base_prompt_count=6, current_n=1
    )
    _, n1_responses, _ = split_final_sequences_at_canonical_prefix(
        n1_full, n1_metadata, n1_collapsed, current_n=1
    )
    assert n1_collapsed == n1_canonical
    assert n1_responses == n1_expected

    equal_canonical = ["<aa>", "<bb>"]
    equal_repeated = [prompt for prompt in equal_canonical for _ in range(2)]
    equal_metadata = [
        {"original_prompt_id": prompt_id, "seq_idx_in_prompt": seq_idx}
        for prompt_id in range(2)
        for seq_idx in range(2)
    ]
    equal_full = [
        equal_canonical[prompt_id] + f"R{prompt_id}{seq_idx}"
        for prompt_id in range(2)
        for seq_idx in range(2)
    ]
    equal_collapsed = canonical_prefixes_from_prompt_major_expansion(
        equal_repeated, base_prompt_count=2, current_n=2
    )
    _, equal_responses, _ = split_final_sequences_at_canonical_prefix(
        equal_full, equal_metadata, equal_collapsed, current_n=2
    )
    assert equal_responses == ["R00", "R01", "R10", "R11"]

    malformed_calls = [
        lambda: canonical_prefixes_from_prompt_major_expansion(
            repeated[:-1], base_prompt_count=6, current_n=5
        ),
        lambda: canonical_prefixes_from_prompt_major_expansion(
            repeated[:7] + ["corrupt"] + repeated[8:],
            base_prompt_count=6,
            current_n=5,
        ),
        lambda: split_final_sequences_at_canonical_prefix(
            full_sequences, [{}] + metadata[1:], collapsed, current_n=5
        ),
        lambda: split_final_sequences_at_canonical_prefix(
            full_sequences,
            [{**metadata[0], "original_prompt_id": 1}] + metadata[1:],
            collapsed,
            current_n=5,
        ),
        lambda: split_final_sequences_at_canonical_prefix(
            ["wrong-prefix" + expected[0]] + full_sequences[1:],
            metadata,
            collapsed,
            current_n=5,
        ),
        lambda: split_final_sequences_at_canonical_prefix(
            full_sequences,
            [{**metadata[0], "seq_idx_in_prompt": 4}] + metadata[1:],
            collapsed,
            current_n=5,
        ),
    ]
    for malformed_call in malformed_calls:
        expect_failure(malformed_call)

    rollout_source = ROLLOUT_PATH.read_text(encoding="utf-8")
    assert "raw_current_prefix_list[" not in rollout_source
    assert rollout_source.count("canonical_prompt_prefixes[") >= 5
    assert locked_function_hashes() == LOCKED_FUNCTION_SHA256
    reward_sha256 = hashlib.sha256(REWARD_PATH.read_bytes()).hexdigest()
    assert reward_sha256 == LOCKED_REWARD_SHA256

    print(
        json.dumps(
            {
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "variable_length_geometry": "6x5",
                "fixed_response_exact_count": len(responses),
                "legacy_response_diff_count": legacy_diff_count,
                "duplicated_prompt_suffix_count": duplicated_suffix_count,
                "dropped_response_head_count": dropped_head_count,
                "otr_replaced_count": sum(replaced_flags),
                "otr_non_replaced_exact_count": len(replaced_flags) - sum(replaced_flags),
                "n1_exact_count": len(n1_responses),
                "equal_length_identity_count": len(equal_responses),
                "malformed_hard_fail_count": len(malformed_calls),
                "locked_otr_reward_search_function_count": len(LOCKED_FUNCTION_SHA256),
                "reward_sha256": reward_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
