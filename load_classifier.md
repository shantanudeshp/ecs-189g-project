## Overview

This guide covers:
- a T4-friendly setup (models + datasets),
- how to build hidden-state datasets,
- how to train the classifier with `cls.py`,
- small experiment suggestions and troubleshooting.

---

## 1) Required files and Knowledge: 

- ### Required Files:
  - `cls.py` (classifier training)
  - `load_classifier.py` (hidden-state dataset builder)
### Why use a classifier?

Correctness is encoded as a **distributed pattern** in hidden states rather than a single value.

The classifier learns a mapping:

```
p(correct) = σ(W·h + b)
```
where:
-   **h** = pooled hidden-state vector
-   **W** = learned weights
-   **σ** = sigmoid

Instead of trusting the LM output directly, we compute:
```
estimated risk of incorrectness = 1 − p(correct)
```

This represents the **likelihood the answer is incorrect**.
The classifier acts as a **probe** that decodes internal reasoning signals.
It learns patterns in hidden states that distinguish correct vs incorrect answers.
Therefore, it can be used to estimate hallucination risk.

----------

### How the paper finds the “strongest layer”

The paper probes where correctness signals emerge:
1.  extract hidden states from all layers
2.  train a classifier per layer
3.  compare accuracy across layers

This reveals **where correctness signals are most strongly encoded**.
---

## 2) T4-friendly models (paper-aligned)

All three are decoder-only LMs, support `output_hidden_states`, and are realistic on a Colab free T4 (16GB) with fp16.

1) TinyLlama (≈1.1B, Llama-family)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Pros: fast, Llama-style closest to paper framing

2) Qwen2.5 0.5B Instruct (≈0.5B)
- `Qwen/Qwen2.5-0.5B-Instruct`
- Pros: extremely light, fast, stable hidden states

3) Gemma 2 2B Instruct (≈2B)
- `google/gemma-2-2b-it`
- Pros: fits T4 fp16, aligns with paper's Gemma family

If gated-model access is an issue, swap #3 with:
- `microsoft/phi-2` (≈2.7B, T4-feasible in fp16)

---

## 3) Datasets (paper-aligned)

The paper uses: NQ-open, MMLU, MedMCQA, GSM8K. On a T4, choose 3 and subsample.

Recommended trio:
- `nq_open` (open-domain Wikipedia QA)
- `medmcqa` (medical MCQ)
- `gsm8k` (math word problems, config `main`)

These give 3 domains for in-domain + cross-domain reporting.

---

## 4) Environment setup (Colab)

Install dependencies:
```bash
pip -q install torch transformers datasets accelerate scikit-learn tqdm
```

Optional:
- Set Hugging Face token in Colab secrets, or run `huggingface-cli login`.

---

## 5) Step A — Build hidden-state datasets

This step:
1) loads dataset split(s),
2) builds k-shot prompts,
3) generates an answer,
4) assigns label = 1 if EM(pred, gold) else 0,
5) extracts hidden states over input prompt tokens,
6) saves `.pth` files for `cls.py`.

Output structure:
```
datasets/<dataset>/<model_name>/hs/<split>/layer_<L>.pth
```

Command template:
```bash
python load_classifier.py \
  --model <HF_MODEL_ID> \
  --dataset <DATASET_NAME> \
  --data-root datasets \
  --k-shot 5 \
  --n-train 2000 --n-val 400 --n-test 400 \
  --layers 0,last \
  --device cuda
```

Example 1: TinyLlama + NQ-open
```bash
python load_classifier.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dataset nq_open \
  --k-shot 5 \
  --n-train 2000 --n-val 400 --n-test 400 \
  --layers 0,last \
  --device cuda
```

Example 2: Qwen 0.5B + MedMCQA
```bash
python load_classifier.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dataset medmcqa \
  --k-shot 5 \
  --n-train 2000 --n-val 400 --n-test 400 \
  --layers 0,last \
  --device cuda
```

Example 3: Gemma2 2B + GSM8K
```bash
python load_classifier.py \
  --model google/gemma-2-2b-it \
  --dataset gsm8k \
  --k-shot 5 \
  --n-train 1200 --n-val 200 --n-test 200 \
  --layers 0,last \
  --device cuda
```

Notes:
- `--layers 0,last` is recommended on T4 to save time/disk.
- The script should print `hidden_size` and `num_layers` needed for classifier training.

---

## Step B — Train classifier with `cls.py`

`cls.py` expects:
- dataset folder: `datasets/<dataset>/<model_name>/hs/...`
- model meta: `--emb-size` and `--num-layers` (unless model is hardcoded)

Important:
- If you saved only `layer_0` and `layer_last`, set `--layers` accordingly.

Command template:
```bash
python cls.py <MODEL_ALIAS> <DATASET_NAME> 0 \
  --data-root datasets \
  --hs-dir hs \
  --layers <LAYER_LIST> \
  --emb-size <HIDDEN_SIZE> \
  --num-layers <NUM_LAYERS> \
  --epochs 20 \
  --batch-size 64 \
  --hidden-size 256 \
  --no-wandb
```

Pooling mode:
- `0` = mean pooling (recommended baseline)

Example (if you saved layers 0 and last only)

Suppose `load_classifier.py` prints:
- hidden_size = 2048
- num_layers = 22
and you saved layers: `0,last` -> `0,22`

Run:
```bash
python cls.py anyname nq_open 0 \
  --data-root datasets \
  --layers 0,22 \
  --emb-size 2048 \
  --num-layers 22 \
  --epochs 20 \
  --batch-size 64 \
  --hidden-size 256 \
  --no-wandb
```

Outputs:
- checkpoints and logs under `clss/<dataset>/<model>/...`

---
