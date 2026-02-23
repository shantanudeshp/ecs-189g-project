#!/usr/bin/env python3
"""
T4-friendly dataset builder
Supports:
- nq_open
- medmcqa
- gsm8k

Outputs hidden states & labels for classifier training.
"""

import argparse
import os
import random
import re
from typing import List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    return s


def exact_match(pred, gold):
    return normalize(pred) == normalize(gold)


def load_nq(split, limit):
    ds = load_dataset("nq_open", split=split)
    data = []
    for row in ds:
        gold = row["answer"][0]
        prompt = f"Question: {row['question']}\nAnswer:"
        data.append((prompt, gold))
    return data[:limit]


def load_medmcqa(split, limit):
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


def load_gsm8k(split, limit):
    ds = load_dataset("gsm8k", "main", split=split)
    data = []
    for r in ds:
        gold = r["answer"].split("####")[-1].strip()
        prompt = f"{r['question']}\nAnswer:"
        data.append((prompt, gold))
    return data[:limit]


def generate_answer(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**inputs, max_new_tokens=8)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("Answer:")[-1].strip()


@torch.no_grad()
def get_hidden_states(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states
    result = []
    for layer in hs:
        tokens = [layer[0, i].cpu().unsqueeze(0) for i in range(layer.shape[1])]
        result.append(tokens)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=400)
    parser.add_argument("--layers", default="0,last")
    parser.add_argument("--k-shot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # resolve device
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU.")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available; falling back to CPU.")
        args.device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float32
        )
        model.to("cpu")
    elif args.device == "mps":
        # MPS works best with float16 or bfloat16 depending on model support
        dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype
        )
        model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.float16,
            device_map="auto"
        )

    loaders = {
        "nq_open": load_nq,
        "medmcqa": load_medmcqa,
        "gsm8k": load_gsm8k,
    }

    load_fn = loaders[args.dataset]

    train = load_fn("train", args.n_train)
    val = load_fn("validation" if args.dataset != "nq_open" else "train", args.n_val)
    test = load_fn("test" if args.dataset != "gsm8k" else "test", args.n_test)

    sample = get_hidden_states(model, tokenizer, train[0][0], args.device)
    num_layers = len(sample) - 1
    hidden_size = sample[0][0].shape[-1]

    print(f"\nModel info:")
    print(f"hidden_size={hidden_size}")
    print(f"num_layers={num_layers}")

    if args.layers == "all":
        layers_to_save = list(range(num_layers + 1))
    else:
        parts = [p.strip() for p in args.layers.split(",") if p.strip()]
        layers_to_save = []
        for p in parts:
            if p == "last":
                layers_to_save.append(num_layers)
            else:
                layers_to_save.append(int(p))

    rng = random.Random(args.seed)

    def format_example(prompt, gold):
        # prompt already ends with "Answer:"
        return f"{prompt} {gold}".strip()

    def build_kshot_prompt(cur_prompt, cur_idx, is_train):
        if args.k_shot <= 0:
            return cur_prompt

        k = args.k_shot
        train_indices = list(range(len(train)))
        if is_train:
            # avoid sampling the current example for train split
            if 0 <= cur_idx < len(train_indices):
                train_indices.pop(cur_idx)

        if k > len(train_indices):
            k = len(train_indices)

        shot_indices = rng.sample(train_indices, k)
        shots = [format_example(train[i][0], train[i][1]) for i in shot_indices]
        return "\n\n".join(shots + [cur_prompt])

    def process(split, name, is_train):
        hs_layers = {l: [] for l in layers_to_save}
        labels = []

        for idx, (prompt, gold) in enumerate(tqdm(split, desc=name)):
            full_prompt = build_kshot_prompt(prompt, idx, is_train)
            pred = generate_answer(model, tokenizer, full_prompt, args.device)
            label = 1 if exact_match(pred, gold) else 0
            hs = get_hidden_states(model, tokenizer, full_prompt, args.device)

            for l in layers_to_save:
                hs_layers[l].append(hs[l])
            labels.append(label)

        return hs_layers, labels

    base = f"datasets/{args.dataset}/{args.model.split('/')[-1]}/hs"
    os.makedirs(base, exist_ok=True)

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        os.makedirs(f"{base}/{split_name}", exist_ok=True)
        hs, labels = process(split_data, split_name, split_name == "train")

        for l in layers_to_save:
            torch.save(
                {"hs": hs[l], "labels": labels},
                f"{base}/{split_name}/layer_{l}.pth"
            )


if __name__ == "__main__":
    main()
