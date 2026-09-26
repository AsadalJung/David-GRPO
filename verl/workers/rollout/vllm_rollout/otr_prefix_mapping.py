"""Fail-closed prompt/response boundary helpers for OTR rollouts."""

from __future__ import annotations

from numbers import Integral
from typing import Any, List, Sequence, Tuple


def canonical_prefixes_from_prompt_major_expansion(
    repeated_prefixes: Sequence[str],
    *,
    base_prompt_count: int,
    current_n: int,
) -> List[str]:
    """Collapse ``[P0 * n, P1 * n, ...]`` after validating every copy."""
    if isinstance(base_prompt_count, bool) or not isinstance(base_prompt_count, Integral):
        raise TypeError("base_prompt_count must be an integer")
    if isinstance(current_n, bool) or not isinstance(current_n, Integral):
        raise TypeError("current_n must be an integer")
    base_prompt_count = int(base_prompt_count)
    current_n = int(current_n)
    if base_prompt_count <= 0:
        raise ValueError(f"base_prompt_count must be positive, got {base_prompt_count}")
    if current_n <= 0:
        raise ValueError(f"current_n must be positive, got {current_n}")

    expected = base_prompt_count * current_n
    if len(repeated_prefixes) != expected:
        raise ValueError(
            "prompt-major expansion length mismatch: "
            f"expected {expected}, got {len(repeated_prefixes)}"
        )

    canonical_prefixes: List[str] = []
    for prompt_id in range(base_prompt_count):
        expanded_index = prompt_id * current_n
        canonical_prefix = repeated_prefixes[expanded_index]
        if not isinstance(canonical_prefix, str):
            raise TypeError(f"prompt prefix {expanded_index} is not text")
        for seq_idx_in_prompt in range(current_n):
            candidate_index = expanded_index + seq_idx_in_prompt
            if repeated_prefixes[candidate_index] != canonical_prefix:
                raise ValueError(
                    "prompt-major copies differ for prompt "
                    f"{prompt_id}: index {candidate_index}"
                )
        canonical_prefixes.append(canonical_prefix)
    return canonical_prefixes


def prompt_id_for_prompt_major_sequence(
    metadata: Any,
    *,
    sequence_index: int,
    base_prompt_count: int,
    current_n: int,
) -> int:
    """Validate metadata and return its canonical per-prompt index."""
    if not isinstance(metadata, dict) or "original_prompt_id" not in metadata:
        raise ValueError(f"missing original_prompt_id at sequence {sequence_index}")
    raw_prompt_id = metadata["original_prompt_id"]
    if isinstance(raw_prompt_id, bool) or not isinstance(raw_prompt_id, Integral):
        raise TypeError(
            f"original_prompt_id must be an integer at sequence {sequence_index}"
        )
    prompt_id = int(raw_prompt_id)
    if prompt_id < 0 or prompt_id >= base_prompt_count:
        raise ValueError(
            f"original_prompt_id {prompt_id} is out of range at sequence {sequence_index}"
        )

    total_sequences = base_prompt_count * current_n
    if sequence_index < 0 or sequence_index >= total_sequences:
        raise ValueError(
            f"sequence index {sequence_index} is out of range for {total_sequences} sequences"
        )
    expected_prompt_id = sequence_index // current_n
    if prompt_id != expected_prompt_id:
        raise ValueError(
            "prompt-major metadata mismatch at sequence "
            f"{sequence_index}: expected prompt {expected_prompt_id}, got {prompt_id}"
        )

    if "seq_idx_in_prompt" in metadata:
        raw_seq_idx = metadata["seq_idx_in_prompt"]
        if isinstance(raw_seq_idx, bool) or not isinstance(raw_seq_idx, Integral):
            raise TypeError(
                f"seq_idx_in_prompt must be an integer at sequence {sequence_index}"
            )
        expected_seq_idx = sequence_index % current_n
        if int(raw_seq_idx) != expected_seq_idx:
            raise ValueError(
                "sequence-within-prompt metadata mismatch at sequence "
                f"{sequence_index}: expected {expected_seq_idx}, got {raw_seq_idx}"
            )
    return prompt_id


def split_final_sequences_at_canonical_prefix(
    final_sequences: Sequence[str],
    supporting_facts: Sequence[Any],
    canonical_prefixes: Sequence[str],
    *,
    current_n: int,
) -> Tuple[List[str], List[str], List[int]]:
    """Split final OTR sequences without consulting the expanded list."""
    base_prompt_count = len(canonical_prefixes)
    expected = base_prompt_count * current_n
    if len(final_sequences) != expected:
        raise ValueError(
            f"final sequence count mismatch: expected {expected}, got {len(final_sequences)}"
        )
    if len(supporting_facts) != expected:
        raise ValueError(
            "final metadata count mismatch: "
            f"expected {expected}, got {len(supporting_facts)}"
        )

    prefixes: List[str] = []
    responses: List[str] = []
    prompt_ids: List[int] = []
    for sequence_index, full_sequence in enumerate(final_sequences):
        if not isinstance(full_sequence, str):
            raise TypeError(f"final sequence {sequence_index} is not text")
        prompt_id = prompt_id_for_prompt_major_sequence(
            supporting_facts[sequence_index],
            sequence_index=sequence_index,
            base_prompt_count=base_prompt_count,
            current_n=current_n,
        )
        canonical_prefix = canonical_prefixes[prompt_id]
        if not full_sequence.startswith(canonical_prefix):
            raise ValueError(
                f"final sequence {sequence_index} does not start with canonical prompt {prompt_id}"
            )
        prefixes.append(canonical_prefix)
        responses.append(full_sequence[len(canonical_prefix) :])
        prompt_ids.append(prompt_id)
    return prefixes, responses, prompt_ids
