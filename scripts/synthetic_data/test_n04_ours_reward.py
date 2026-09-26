#!/usr/bin/env python3
"""Regression contract for the byte-exact canonical Ours reward semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path


REWARD_PATH = (
    Path(__file__).resolve().parents[2]
    / "verl/utils/reward_score/hotpotqa.py"
)
SPEC = importlib.util.spec_from_file_location("canonical_ours_reward", REWARD_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load canonical Ours reward: {REWARD_PATH}")
REWARD_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REWARD_MODULE)
compute_score = REWARD_MODULE.compute_score


FACTS_ONE = {"title": ["Doc A"], "evidence_total_count": 1}
FACTS_TWO = {"title": ["Doc A", "Doc B"], "evidence_total_count": 2}


def score(
    solution: str,
    *,
    training: bool,
    facts: dict = FACTS_ONE,
    answer: str = "yes",
) -> float:
    return compute_score(
        solution,
        answer,
        method="strict",
        extra_info={
            "supporting_facts": facts,
            "trainer_config": {
                "give_partial_reward": training,
                "require_search_match_for_answer": False,
                "use_answer_in_search_reward": False,
                "partial_reward_weight": 0.5,
            },
        },
    )


def trace(
    query: str = "query",
    title: str = "Doc A",
    answer: str = "yes",
) -> str:
    return (
        "<think>"
        f"<begin_search>{query}</end_search>"
        f'<search_result>result 1: "{title}"</search_result>'
        "</think>"
        f"\\boxed{{{answer}}}"
    )


def assert_scores(
    solution: str,
    expected_training: float,
    expected_validation: float,
    *,
    facts: dict = FACTS_ONE,
) -> None:
    actual_training = score(solution, training=True, facts=facts)
    actual_validation = score(solution, training=False, facts=facts)
    assert actual_training == expected_training, (
        solution,
        actual_training,
        expected_training,
    )
    assert actual_validation == expected_validation, (
        solution,
        actual_validation,
        expected_validation,
    )


def main() -> None:
    valid = trace()
    assert_scores(valid, 1.0, 1.0)

    assert_scores(valid.replace("</think>", ""), 0.0, 0.0)
    assert_scores(
        valid
        + "<begin_search>late</end_search>"
        + '<search_result>result 1: "Doc A"</search_result>',
        0.0,
        0.0,
    )
    assert_scores("<think>reasoning</think>\\boxed{yes}", 0.0, 0.0)
    assert_scores(valid.replace("\\boxed{yes}", "yes"), 0.0, 0.0)
    assert_scores(valid.replace("\\boxed{yes}", "\\boxed{}"), 0.75, 0.5)

    unbalanced = valid.replace("</search_result>", "")
    assert_scores(unbalanced, 0.0, 1.0)

    duplicate = (
        "<think>"
        "<begin_search>query</end_search>"
        '<search_result>result 1: "Doc A"</search_result>'
        "<begin_search>query</end_search>"
        '<search_result>result 1: "Doc A"</search_result>'
        "</think>"
        "\\boxed{yes}"
    )
    assert_scores(duplicate, 0.0, 1.0)

    whitespace_only_duplicate = duplicate.replace(
        "<begin_search>query</end_search>"
        '<search_result>result 1: "Doc A"</search_result>'
        "</think>",
        "<begin_search> query </end_search>"
        '<search_result>result 1: "Doc A"</search_result>'
        "</think>",
    )
    assert_scores(whitespace_only_duplicate, 1.0, 1.0)

    assert_scores(trace(title="Irrelevant"), 0.5, 1.0)
    assert_scores(trace(), 0.75, 1.0, facts=FACTS_TWO)

    print(
        "canonical_ours_reward=validated; "
        "training_hard_gates=enabled; "
        "validation_legacy_leniency=enabled; "
        "require_search_match_for_answer=false"
    )


if __name__ == "__main__":
    main()
