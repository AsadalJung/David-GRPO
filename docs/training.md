# Training implementation

## What changed

1. OTR prompt prefixes are indexed by the original question, not an index into
   the rollout-expanded list. Prompt-major metadata and final boundaries are
   checked before splitting prompts and responses.
2. The corrected actor passes the trainer's `loss_mask` through PPO updates.
   Observations remain in the model context but do not contribute to PG, entropy,
   KL, or their action-token denominators. Masking precedes nonlinear KL/ratio
   operations so excluded positions do not inject non-finite gradients.
3. Canonical question UIDs survive batch reordering and DP/SP/microbatch splitting.
   The corrected reduction takes token means, then sequence means within each
   question, then a question mean over the global optimizer minibatch. Microbatch
   contributions are summed without a second gradient-accumulation division.
4. Main training uses the `mse` (k2) KL term: with
   `d = log pi - log pi_ref`, it returns `0.5 * d**2`, with coefficient 0.001.
   This differs from the historical k1 loss `d`; the release does not claim
   isolated performance gains from the estimator change.

## Main-training configuration

| Setting | Main training |
| --- | --- |
| Actor action-only mask | on |
| Prompt-balanced actor reduction | on |
| KL loss type | `mse` / k2 |
| KL coefficient | 0.001 |
| LR | 1e-6 |
| LR warmup / schedule | 0 / constant |
| OTR prefix alignment fix | on |
| OTR / partial reward / trainer observation mask | on |

There is a single corrected main-training configuration, not selectable legacy
and corrected profiles. Its KL-enable/type and actor masking/balancing arguments
are applied last in the launcher so other model-specific overrides cannot
silently disable those corrections. The shared library still contains generic
KL utilities; the main launcher does not expose them as alternate profiles.

The warmup launcher explicitly retains the historical actor-loss convention and
has KL loss disabled. Main-training defaults do not silently change warmup.

## Model/data-specific settings

The main launcher is a four-GPU template: TP1/SP2, train batch24, rollout n5,
PPO mini12, actor micro2, rollout log-prob micro2, reference log-prob micro4,
prompt1024/response8192, temperature0.6, max search10, 215 steps, save/test10.
It leaves actor, optimizer and reference offload disabled. W&B defaults to offline.
These settings are not a claim that all backbone runs used identical microbatches,
PPO minibatches, GPU counts, warmup checkpoints, or decoding temperatures.

Use environment variables for paths, retrieval endpoints, GPU visibility,
save/test cadence and experiment name. Additional Hydra arguments allow explicit
model-specific overrides, except for the fixed corrected actor settings. For example:

```bash
TRAIN_DATA_PATH=/path/to/constructed-train.parquet \
VAL_DATA_PATH=/path/to/fixed-validation.parquet \
WARMUP_CHECKPOINT_PATH=/path/to/expert-seeded-checkpoint \
RETRIEVER_URL=http://localhost:8011/retrieve \
VAL_RETRIEVER_URL=http://localhost:8001/retrieve \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash scripts/train.sh actor_rollout_ref.rollout.gpu_memory_utilization=0.75
```

For annotated-evidence data, use its matching corpus (normally port 8001) for
training as well. Do not substitute synthetic rows without the associated retrieval
documents. The release retains the existing annotated-evidence training (5,276),
validation (400), warmup (4) and evaluation (23,078) Parquet rows. It does not
include synthetic training Parquet, retrieval corpora/indices or checkpoints.
The [original synthetic JSONL](../data/synthetic/README.md) is provided separately:
5,276 distinct QA instances, not the later oversampled derivatives.
Data generation and exact per-result run manifests are not provided by this template.
In particular, the current figure/table report manuscript results, not results of
a new run of these defaults. No blanket claim is made that historical results used
the corrected actor implementation.

The pinned baseline environment uses Transformers 4.47.1 and vLLM 0.6.3. It is not
a turnkey Qwen3 environment: Qwen3 requires a compatible model stack and rollout
adapter. Do not claim the base requirements alone reproduce Qwen3 experiments.

## CPU regression checks

In a compatible training environment, from the repository root:

```bash
export CUDA_VISIBLE_DEVICES=
export PYTHONPATH=.
python -m unittest discover -s tests -p 'test_*contract.py' -v
python scripts/synthetic_data/test_otr_prefix_mapping.py
python scripts/synthetic_data/test_retriever_routing.py
python scripts/synthetic_data/test_validation_outputs.py
python scripts/synthetic_data/test_n04_ours_reward.py
```

The actor tests use a tiny differentiable CPU toy model; they do not start model
training, retrieval services or GPU jobs. Distributed tests use four CPU/Gloo ranks.
They check masking, finite gradients, UID grouping, global reductions and invariance
to microbatch partitioning, not downstream benchmark performance.
