#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from verl.workers.rollout.vllm_rollout import vllm_rollout_coa as rollout


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def load_rollout_config(env):
    config_path = (
        Path(__file__).resolve().parents[2]
        / "verl"
        / "trainer"
        / "config"
        / "ppo_trainer.yaml"
    )
    with patch.dict(os.environ, env, clear=True):
        config = OmegaConf.load(config_path)
        return OmegaConf.to_container(
            config.actor_rollout_ref.rollout,
            resolve=True,
        )


def autocoa_payload():
    return [
        {
            "documents": [
                {"contents": '"France"\nbody'},
                {"contents": '"Paris"\nbody'},
                {"contents": '"Toulouse"\nbody'},
            ],
            "scores": [1.0, 0.9, 0.8],
        }
    ]


def tree_payload():
    return {
        "result": [
            [
                {"document": {"contents": '"France"\nbody'}, "score": 1.0},
                {"document": {"contents": '"Paris"\nbody'}, "score": 0.9},
                {"document": {"contents": '"Toulouse"\nbody'}, "score": 0.8},
            ]
        ]
    }


def main():
    split = load_rollout_config(
        {
            "RETRIEVER_URL": "http://127.0.0.1:8011/retrieve",
            "VAL_RETRIEVER_URL": "http://127.0.0.1:8001/retrieve",
        }
    )
    assert split["retriever_url"] == "http://127.0.0.1:8011/retrieve"
    assert split["validation_retriever_url"] == "http://127.0.0.1:8001/retrieve"

    fallback = load_rollout_config(
        {"RETRIEVER_URL": "http://127.0.0.1:8011/retrieve"}
    )
    assert fallback["validation_retriever_url"] == fallback["retriever_url"]
    assert rollout._select_retriever_url(
        False,
        split["retriever_url"],
        split["validation_retriever_url"],
    ).endswith(":8011/retrieve")
    assert rollout._select_retriever_url(
        True,
        split["retriever_url"],
        split["validation_retriever_url"],
    ).endswith(":8001/retrieve")

    for url in (
        "http://127.0.0.1:8011/retrieve",
        "http://127.0.0.1:8001/retrieve",
    ):
        with patch.object(
            rollout.requests,
            "post",
            return_value=FakeResponse(autocoa_payload()),
        ) as post:
            rendered = rollout.get_search_results(["capital"], retriever_url=url)
        post.assert_called_once()
        assert post.call_args.args[0] == url
        assert post.call_args.kwargs["json"] == {
            "query": ["capital"],
            "tok_k": 3,
            "return_score": True,
        }
        assert '"France"' in rendered[0]

    tree_url = "http://127.0.0.1:8003/retrieve"
    with patch.object(
        rollout.requests,
        "post",
        return_value=FakeResponse(tree_payload()),
    ) as post:
        rendered = rollout.get_search_results(["capital"], retriever_url=tree_url)
    assert post.call_args.args[0] == tree_url
    assert post.call_args.kwargs["json"] == {
        "queries": ["capital"],
        "topk": 3,
        "return_scores": True,
    }
    assert '"France"' in rendered[0]

    print("retriever_routing_tests=ok")


if __name__ == "__main__":
    main()
