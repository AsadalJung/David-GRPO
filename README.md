<div align="center">
  <img src="resources/logo.png" width="380px">
</div>
<h1 align="center" style="margin-top: 10px;">Can David Beat Goliath? On Multi-Hop Reasoning with Resource-Constrained Agents</h1>

<p align="center">
  Hojae Han<sup>1</sup>,
  Heeyun Jung<sup>2</sup>,
  Jongyoon Kim<sup>3</sup>,
  Seung-won Hwang<sup>3*</sup>
  <br>
  <sup>1</sup>ETRI &nbsp;&nbsp;
  <sup>2</sup>HKU &nbsp;&nbsp;
  <sup>3</sup>SNU
  <br>
  <sup>*</sup>Corresponding author
</p>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2601.21699)
[![PDF](https://img.shields.io/badge/PDF-2601.21699-blue?style=flat-square)](https://arxiv.org/pdf/2601.21699)

</div>

## News
- [Jan 30, 2026]: Codebase prepared for release.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Overview
DAVID-GRPO enables **multi-hop reasoning under tight compute budgets** by reinterpreting RL failures as a **cold-start retrieval** problem and adapting classic IR principles. The method combines:
- **Few-shot warm-start** (mixed off-/on-policy GRPO)
- **Grounded retrieval reward** (evidence recall across the full trajectory)
- **Grounded expansion** (resampling truncated near-miss trajectories)

<p align="center">
  <img alt="Figure 2. Overview of DAVID-GRPO" src="resources/main.png" />
  <br>
  <em>Figure 2. Overview of DAVID-GRPO.</em>
</p>

## Results
Numbers are Exact Match (EM) / F1. Dashes indicate not reported. † are reported from Ji et al. (2025a).

### Qwen2.5-1.5B
<table>
  <thead>
    <tr>
      <th align="left">Method</th>
      <th>HotpotQA</th>
      <th>2Wiki</th>
      <th>Musique</th>
      <th>Bamboogle</th>
      <th>Bamtwoogle</th>
      <th>Antileak-m</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Direct Inference†</td>
      <td>5.9 / -</td><td>4.3 / -</td><td>2.6 / -</td><td>8.0 / -</td><td>- / -</td><td>- / -</td><td>- / -</td>
    </tr>
    <tr>
      <td align="left">Search-o1†</td>
      <td>11.6 / -</td><td>12.2 / -</td><td>3.1 / -</td><td>13.0 / -</td><td>- / -</td><td>- / -</td><td>- / -</td>
    </tr>
    <tr>
      <td align="left">GRPO†</td>
      <td>14.6 / -</td><td>24.4 / -</td><td>2.2 / -</td><td>4.0 / -</td><td>- / -</td><td>- / -</td><td>- / -</td>
    </tr>
    <tr>
      <td align="left">Tree-GRPO† (High)</td>
      <td>29.5 / -</td><td>26.8 / -</td><td>6.6 / -</td><td>13.6 / -</td><td>- / -</td><td>- / -</td><td>- / -</td>
    </tr>
    <tr>
      <td align="left">Tree-GRPO</td>
      <td>12.9 / 18.6</td><td>20.5 / 23.3</td><td>2.1 / 7.2</td><td>12.0 / 15.5</td><td>5.0 / 8.2</td><td>11.7 / 15.7</td><td>10.7 / 14.8</td>
    </tr>
    <tr>
      <td align="left">StepSearch</td>
      <td>11.9 / 18.1</td><td>13.5 / 18.1</td><td>2.2 / 6.9</td><td>3.2 / 8.7</td><td>4.0 / 6.0</td><td>12.1 / 17.3</td><td>7.8 / 12.5</td>
    </tr>
    <tr>
      <td align="left">Search-R1-v0.3 (retrieval reward)</td>
      <td>19.0 / 26.5</td><td>21.3 / 26.4</td><td>3.6 / 9.2</td><td>8.0 / 14.3</td><td>3.0 / 5.5</td><td>16.7 / 22.2</td><td>11.9 / 17.4</td>
    </tr>
    <tr>
      <td align="left"><strong>DAVID-GRPO</strong></td>
      <td><strong>24.8 / 33.8</strong></td><td><strong>27.2 / 32.3</strong></td><td><strong>7.1 / 12.6</strong></td><td><strong>14.4 / 24.2</strong></td><td><strong>22.0 / 25.4</strong></td><td><strong>36.3 / 41.1</strong></td><td><strong>22.0 / 28.2</strong></td>
    </tr>
  </tbody>
</table>

### Llama-3.2-1B
<table>
  <thead>
    <tr>
      <th align="left">Method</th>
      <th>HotpotQA</th>
      <th>2Wiki</th>
      <th>Musique</th>
      <th>Bamboogle</th>
      <th>Bamtwoogle</th>
      <th>Antileak-m</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Tree-GRPO</td>
      <td>12.4 / 18.4</td><td>20.5 / 23.4</td><td>1.6 / 7.1</td><td>4.0 / 8.9</td><td>9.0 / 11.9</td><td>16.0 / 20.7</td><td>10.6 / 15.1</td>
    </tr>
    <tr>
      <td align="left">StepSearch</td>
      <td>16.0 / 24.5</td><td>11.7 / 16.2</td><td>2.7 / 9.7</td><td>4.0 / 11.0</td><td>3.0 / 5.7</td><td>22.4 / 31.4</td><td>10.0 / 16.4</td>
    </tr>
    <tr>
      <td align="left">Search-R1-v0.3 (retrieval reward)</td>
      <td>8.5 / 12.4</td><td>13.5 / 16.2</td><td>0.7 / 3.7</td><td>0.8 / 1.6</td><td>2.0 / 4.7</td><td>13.0 / 15.3</td><td>6.4 / 9.0</td>
    </tr>
    <tr>
      <td align="left"><strong>DAVID-GRPO</strong></td>
      <td><strong>17.7 / 25.2</strong></td><td><strong>16.1 / 21.4</strong></td><td><strong>3.2 / 8.5</strong></td><td><strong>8.0 / 14.5</strong></td><td><strong>3.0 / 6.6</strong></td><td><strong>23.3 / 31.7</strong></td><td><strong>11.9 / 18.0</strong></td>
    </tr>
  </tbody>
</table>

### Qwen2.5-0.5B
<table>
  <thead>
    <tr>
      <th align="left">Method</th>
      <th>HotpotQA</th>
      <th>2Wiki</th>
      <th>Musique</th>
      <th>Bamboogle</th>
      <th>Bamtwoogle</th>
      <th>Antileak-m</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">Tree-GRPO</td>
      <td>9.2 / 12.8</td><td>20.9 / 22.8</td><td>0.7 / 3.8</td><td>3.2 / 5.2</td><td>4.0 / 5.5</td><td>10.3 / 12.7</td><td>8.1 / 10.5</td>
    </tr>
    <tr>
      <td align="left">StepSearch</td>
      <td>2.1 / 4.5</td><td>4.3 / 7.3</td><td>0.2 / 1.8</td><td>1.6 / 2.1</td><td>0.0 / 0.6</td><td>0.2 / 0.8</td><td>1.4 / 2.9</td>
    </tr>
    <tr>
      <td align="left">Search-R1-v0.3 (retrieval reward)</td>
      <td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td><td>0.0 / 0.0</td>
    </tr>
    <tr>
      <td align="left"><strong>DAVID-GRPO</strong></td>
      <td><strong>10.8 / 16.0</strong></td><td><strong>17.4 / 20.8</strong></td><td><strong>2.0 / 5.3</strong></td><td><strong>4.8 / 8.1</strong></td><td><strong>6.0 / 7.0</strong></td><td><strong>10.6 / 14.4</strong></td><td><strong>8.6 / 11.9</strong></td>
    </tr>
  </tbody>
</table>

## Quick Start
Set your repo root once:
```bash
export REPO_ROOT=/path/to/David-GRPO
```

### RL Environment (verl)
```bash
conda create -n david-grpo python=3.10 -y
conda activate david-grpo
pip install -r ${REPO_ROOT}/requirements.txt
```

### HotpotQA Retriever (training)
```bash
conda create -n hotpot-retriever python=3.10 -y
conda activate hotpot-retriever
pip install -r ${REPO_ROOT}/hotpotqa_retriever/requirements.txt
```
Build the HotpotQA corpus (AutoCoA-compatible) from train+dev JSONL:
```bash
python ${REPO_ROOT}/hotpotqa_retriever/build_hotpotqa_corpus.py \
  --train /path/to/hotpotqa_train.jsonl \
  --dev /path/to/hotpotqa_dev.jsonl \
  --output ${REPO_ROOT}/hotpotqa_retriever/hotpotqa_corpus.json
```
Then create the FAISS index:
```bash
python ${REPO_ROOT}/hotpotqa_retriever/create_faiss_index.py
```
Launch:
```bash
conda activate hotpot-retriever
python ${REPO_ROOT}/hotpotqa_retriever/run_server.py --port 8001
```
`train.sh` expects `RETRIEVER_URL=http://localhost:8001/retrieve`.

### Eval Retriever (wiki-18 / wiki-24)
```bash
conda create -n eval-retriever python=3.10 -y
conda activate eval-retriever
pip install -r ${REPO_ROOT}/eval_retriever/requirements.txt
```

#### Prepare wiki-18 (public, prebuilt index)
```bash
${REPO_ROOT}/eval_retriever/scripts/prepare_wiki18.sh
```
Creates:
```
${REPO_ROOT}/data/wiki/e5_Flat.index
${REPO_ROOT}/data/wiki/wiki-18.jsonl
```

#### Prepare wiki-24 (your own dump)
1) Place your JSONL:
```
${REPO_ROOT}/data/wiki-24/wiki-24.jsonl
```
2) Build FAISS index:
```bash
python ${REPO_ROOT}/eval_retriever/scripts/build_faiss_index.py \
  --corpus ${REPO_ROOT}/data/wiki-24/wiki-24.jsonl \
  --output ${REPO_ROOT}/data/wiki-24/index/e5_Flat.index \
  --model intfloat/e5-base-v2 \
  --batch-size 256
```

#### Launch eval retrievers
```bash
# wiki-18 (standard eval)
${REPO_ROOT}/eval_retriever/local_retrieval_launch.sh

# wiki-24 (AntiLeakBench)
${REPO_ROOT}/eval_retriever/local_retrieval_launch_wiki24.sh
```
Set `RETRIEVER_URL` accordingly:
- wiki-18: `http://localhost:8003/retrieve`
- wiki-24: `http://localhost:8004/retrieve`

## Training
Main training script:
```bash
${REPO_ROOT}/scripts/train.sh
```
Notes:
- Set `WARMUP_CHECKPOINT_PATH` to your warmup checkpoint.
- Ensure the HotpotQA retriever is running on port 8001.

## Evaluation
The generation script expects the eval retriever on 8003 (wiki-18) or 8004 (wiki-24):
```bash
${REPO_ROOT}/scripts/eval/generate_response.sh
```

## Acknowledgements
- Codebase derived from AutoCoA and Search-R1.
- Reinforcement learning pipeline uses the **verl** framework.

References:
- https://github.com/ADaM-BJTU/AutoCoA
- https://github.com/PeterGriffinJin/Search-R1
- https://github.com/verl-project/verl

## Citation
```bibtex
@article{han2026davidgrpo,
  title={Can David Beat Goliath? On Multi-Hop Reasoning with Resource-Constrained Agents},
  author={Hojae Han and Heeyun Jung and Jongyoon Kim and Seung-won Hwang},
  journal={arXiv preprint arXiv:2601.21699},
  year={2026}
}
```

## License
This project is licensed under the Apache-2.0 License. See `LICENSE` for details.
