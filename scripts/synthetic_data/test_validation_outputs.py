#!/usr/bin/env python3
"""Dependency-light smoke tests for complete validation answer persistence."""

import json
import tempfile
from pathlib import Path

from verl.trainer.validation_output import (
    ValidationOutputWriter,
    build_validation_record,
    extract_search_metadata,
    is_primary_process,
    resolve_validation_output_base_dir,
)


def test_record_building() -> dict:
    response = (
        "<think><begin_search>Herman Wouk novelist</end_search>\n"
        '<search_result>result 1: \"Herman Wouk\"\\nA novelist.</search_result>'
        "</think>\\boxed{Herman Wouk}"
    )
    metadata = {
        "_id": "hotpot-row-한글",
        "index": 12,
        "data_source": "HotpotQA",
        "raw_prompt": [
            {"role": "system", "content": "Answer carefully."},
            {
                "role": "user",
                "content": (
                    "Tool instructions.\n\n"
                    "Now, please answer the user's question below:\n"
                    "Who wrote the novel?"
                ),
            },
        ],
        "reward_model": {"style": "rule", "ground_truth": "Herman Wouk"},
        "reward_components": {"answer": 1.0},
        "supporting_facts": {"title": ["Herman Wouk"]},
        "extra_info": {"index": 12, "split": "dev"},
    }
    record = build_validation_record(
        validation_row=0,
        prompt="rendered prompt",
        response=response,
        total_reward=1.0,
        example_metadata=metadata,
    )
    assert record["row_id"] == "hotpot-row-한글"
    assert record["prompt_id"] == 12
    assert record["question"] == "Who wrote the novel?"
    assert record["gold_answer"] == "Herman Wouk"
    assert record["full_raw_response"] == response
    assert record["extracted_answer"] == "herman wouk"
    assert record["total_reward"] == 1.0
    assert record["reward_components"] == {"answer": 1.0}
    assert record["supporting_facts"] == {"title": ["Herman Wouk"]}
    assert record["search_metadata"]["queries"] == ["Herman Wouk novelist"]
    assert record["search_metadata"]["retrieved_titles"] == ["Herman Wouk"]
    return record


def test_atomic_unique_step_files(record: dict) -> None:
    with tempfile.TemporaryDirectory(prefix="validation-output-smoke-") as temporary_dir:
        base_dir = Path(temporary_dir) / "validation_outputs"
        writer = ValidationOutputWriter(
            base_dir,
            session_metadata={"experiment_name": "smoke"},
        )
        first_path = writer.write_step(global_step=23, records=[record])
        second_path = writer.write_step(global_step=23, records=[record])

        assert first_path != second_path
        assert first_path.is_file()
        assert second_path.is_file()
        assert first_path.parent == second_path.parent == writer.session_dir
        assert (writer.session_dir / "_session.json").is_file()
        assert not list(writer.session_dir.glob("*.tmp"))
        assert not list(writer.session_dir.glob(".*.tmp"))

        with first_path.open(encoding="utf-8") as handle:
            first_payload = json.loads(handle.readline())
            assert handle.readline() == ""
        with second_path.open(encoding="utf-8") as handle:
            second_payload = json.loads(handle.readline())

        assert first_payload["schema"] == "verl-validation-answer/v1"
        assert first_payload["global_step"] == 23
        assert first_payload["validation_call"] == 1
        assert second_payload["global_step"] == 23
        assert second_payload["validation_call"] == 2
        assert first_payload["validation_session_id"] == writer.session_id
        assert second_payload["validation_session_id"] == writer.session_id


def test_configuration_and_rank_guards() -> None:
    assert (
        resolve_validation_output_base_dir("/run/checkpoints")
        == Path("/run/checkpoints/validation_outputs")
    )
    assert (
        resolve_validation_output_base_dir("/run/checkpoints", "/run/answers")
        == Path("/run/answers")
    )
    assert is_primary_process({}) is True
    assert is_primary_process({"RANK": "0"}) is True
    assert is_primary_process({"RANK": "1"}) is False
    assert is_primary_process({"SLURM_PROCID": "not-an-integer"}) is False

    metadata = extract_search_metadata("no tool call")
    assert metadata["query_count"] == 0
    assert metadata["result_blocks"] == []


def main() -> None:
    record = test_record_building()
    test_atomic_unique_step_files(record)
    test_configuration_and_rank_guards()
    print("validation output smoke tests passed")


if __name__ == "__main__":
    main()
