#!/usr/bin/env python3
"""
Build intervention training data: (hs, hs_fct) pairs per layer.

hs      = hidden states over (prompt + model_answer)
hs_fct  = if correct -> hs; else -> hidden states over (prompt + gold_answer)

Matches prompt formatting + gold extraction from your current load_classifier.py.
"""

import argparse
import os
import random
import re
from typing import List, Tuple, Dict, Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def extract_last_number(text: str):
    matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def gsm8k_match(pred: str, gold: str) -> bool:
    pred_num = extract_last_number(pred)
    gold_num = extract_last_number(gold)
    if pred_num is None or gold_num is None:
        return exact_match(pred, gold)
    return pred_num == gold_num


def load_nq(split: str, limit: int) -> List[Tuple[str, str]]:
    ds = load_dataset("nq_open", split=split)
    data = []
    for row in ds:
        gold = row["answer"][0]
        prompt = f"Question: {row['question']}\nAnswer:"
        data.append((prompt, gold))
    return data[:limit]


def load_medmcqa(split: str, limit: int) -> List[Tuple[str, str]]:
    ds = load_dataset("medmcqa", split=split)
    data = []
    for r in ds:
        choices = [r["opa"], r["opb"], r["opc"], r["opd"]]
        gold = choices[r["cop"]]
        prompt = (
            f"Question: {r['question']}\n"
            f"A. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n"
            "Answer:"
        )
        data.append((prompt, gold))
    return data[:limit]


def load_gsm8k(split: str, limit: int) -> List[Tuple[str, str]]:
    ds = load_dataset("gsm8k", "main", split=split)
    data = []
    for r in ds:
        gold = r["answer"].split("####")[-1].strip()
        prompt = f"{r['question']}\nAnswer:"
        data.append((prompt, gold))
    return data[:limit]


def generate_answer(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("Answer:")[-1].strip()


@torch.no_grad()
def get_hidden_states(model, tokenizer, text: str, device: str):
    """
    Returns: list over layers; each layer is list of token tensors [1, d]
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states
    result = []
    for layer in hs:
        tokens = [layer[0, i].detach().cpu().unsqueeze(0) for i in range(layer.shape[1])]
        result.append(tokens)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model id (e.g., distilgpt2, gpt2)")
    p.add_argument("--dataset", required=True, choices=["nq_open", "medmcqa", "gsm8k"])
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=400)
    p.add_argument("--n-test", type=int, default=400)
    p.add_argument("--layers", default="0,last")
    p.add_argument("--k-shot", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available; falling back to CPU.")
        args.device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        model.to("cpu")
    elif args.device == "mps":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
        model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")

    loaders = {"nq_open": load_nq, "medmcqa": load_medmcqa, "gsm8k": load_gsm8k}
    load_fn = loaders[args.dataset]

    train = load_fn("train", args.n_train)
    if args.dataset in ("gsm8k", "nq_open"):
        val = load_fn("train", args.n_val)
    else:
        val = load_fn("validation", args.n_val)
    # Some datasets (e.g., nq_open) don't expose a "test" split on HF; fall back to validation.
    test_split = "test"
    if args.dataset == "nq_open":
        test_split = "validation"
    test = load_fn(test_split, args.n_test)

    # determine layer indices
    sample = get_hidden_states(model, tokenizer, train[0][0], args.device)
    num_layers = len(sample) - 1
    if args.layers == "all":
        layers_to_save = list(range(num_layers + 1))
    else:
        parts = [x.strip() for x in args.layers.split(",") if x.strip()]
        layers_to_save = []
        for x in parts:
            if x == "last":
                layers_to_save.append(num_layers)
            else:
                layers_to_save.append(int(x))

    rng = random.Random(args.seed)

    def format_example(prompt: str, gold: str) -> str:
        return f"{prompt} {gold}".strip()

    def build_kshot_prompt(cur_prompt: str, cur_idx: int, is_train: bool) -> str:
        if args.k_shot <= 0:
            return cur_prompt
        k = args.k_shot
        train_indices = list(range(len(train)))
        if is_train and 0 <= cur_idx < len(train_indices):
            train_indices.pop(cur_idx)
        if k > len(train_indices):
            k = len(train_indices)
        shot_indices = rng.sample(train_indices, k)
        shots = [format_example(train[i][0], train[i][1]) for i in shot_indices]
        return "\n\n".join(shots + [cur_prompt])

    def process(split_data, split_name: str, is_train: bool):
        hs_layers: Dict[int, List[Any]] = {l: [] for l in layers_to_save}
        hs_fct_layers: Dict[int, List[Any]] = {l: [] for l in layers_to_save}
        meta: List[Dict[str, Any]] = []

        for idx, (prompt, gold) in enumerate(tqdm(split_data, desc=f"{split_name}_fct")):
            full_prompt = build_kshot_prompt(prompt, idx, is_train)

            pred = generate_answer(model, tokenizer, full_prompt, args.device, args.max_new_tokens)
            if args.dataset == "gsm8k":
                correct = gsm8k_match(pred, gold)
            else:
                correct = exact_match(pred, gold)

            # build concatenated texts
            text_pred = f"{full_prompt} {pred}".strip()
            text_gold = f"{full_prompt} {gold}".strip()

            hs_pred = get_hidden_states(model, tokenizer, text_pred, args.device)
            if correct:
                hs_target = hs_pred
            else:
                hs_target = get_hidden_states(model, tokenizer, text_gold, args.device)

            for l in layers_to_save:
                hs_layers[l].append(hs_pred[l])
                hs_fct_layers[l].append(hs_target[l])

            meta.append({
                "prompt": full_prompt,
                "gold": gold,
                "pred": pred,
                "correct": bool(correct),
            })

        return hs_layers, hs_fct_layers, meta

    base = f"datasets/{args.dataset}/{args.model.split('/')[-1]}/hs"
    for split_name, split_data, is_train in [
        ("train_fct", train, True),
        ("val_fct", val, False),
        ("test_fct", test, False),
    ]:
        out_dir = os.path.join(base, split_name)
        os.makedirs(out_dir, exist_ok=True)
        hs, hs_fct, meta = process(split_data, split_name, is_train)

        for l in layers_to_save:
            torch.save(
                {"hs": hs[l], "hs_fct": hs_fct[l], "meta": meta},
                os.path.join(out_dir, f"layer_{l}.pth")
            )

    print("Done.")


if __name__ == "__main__":
    main()
