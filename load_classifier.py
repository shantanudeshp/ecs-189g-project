#!/usr/bin/env python3
import argparse
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


# -------------------------
# Dataset loaders
# -------------------------

def load_nq(split: str, n: int) -> List[Tuple[str, str]]:
    """
    Returns list of (prompt, gold_answer).
    nq_open on HF is often "nq_open" / "nq_open" depending on hub naming.
    We'll try common variants.
    """
    ds = None
    for name in ["nq_open", "nq_open"]:
        try:
            ds = load_dataset(name, split=split)
            break
        except Exception:
            continue
    if ds is None:
        # fallback: some mirrors use "nq_open" with config "default"
        ds = load_dataset("nq_open", split=split)

    out = []
    for ex in ds:
        q = ex.get("question") or ex.get("input") or ex.get("query")
        # answers may be list
        a = ex.get("answer") or ex.get("answers")
        if isinstance(a, list):
            gold = a[0] if a else ""
        else:
            gold = a if a is not None else ""
        prompt = f"Question: {q}\nAnswer:"
        out.append((prompt, str(gold)))
        if len(out) >= n:
            break
    return out


def load_gsm8k(split: str, n: int) -> List[Tuple[str, str]]:
    ds = load_dataset("gsm8k", "main", split=split)
    out = []
    for ex in ds:
        q = ex["question"]
        # gold is in "answer" with "#### <number>"
        gold = ex["answer"]
        prompt = f"Question: {q}\nAnswer:"
        out.append((prompt, gold))
        if len(out) >= n:
            break
    return out


def load_medmcqa(split: str, n: int) -> List[Tuple[str, str]]:
    ds = load_dataset("medmcqa", split=split)
    out = []
    for ex in ds:
        q = ex["question"]
        opa = ex.get("opa", "")
        opb = ex.get("opb", "")
        opc = ex.get("opc", "")
        opd = ex.get("opd", "")
        # correct option index often in "cop" (0/1/2/3)
        cop = ex.get("cop", None)
        gold_letter = None
        if cop is not None:
            try:
                cop_i = int(cop)
                gold_letter = ["A", "B", "C", "D"][cop_i]
            except Exception:
                gold_letter = None

        prompt = (
            f"Question: {q}\n"
            f"A) {opa}\n"
            f"B) {opb}\n"
            f"C) {opc}\n"
            f"D) {opd}\n"
            f"Answer:"
        )
        # For gold string, keep letter if available; else store the option text if present.
        if gold_letter is not None:
            gold = gold_letter
        else:
            # fallback: try answer key fields
            gold = str(ex.get("answer", "")).strip()
        out.append((prompt, gold))
        if len(out) >= n:
            break
    return out


# -------------------------
# Normalization + matching
# -------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+", re.UNICODE)

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_text(pred) == normalize_text(gold)


def relaxed_match(pred: str, gold: str) -> bool:
    """
    Relaxed textual match for short answers:
    - normalized substring either way
    - falls back to exact match
    """
    p = normalize_text(pred)
    g = normalize_text(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    if g in p or p in g:
        return True
    return False


_NUM_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?")

def extract_last_number(s: str) -> str:
    """
    Extract last numeric token from string; returns normalized number string (no commas).
    """
    nums = _NUM_RE.findall(s)
    if not nums:
        return ""
    last = nums[-1].replace(",", "")
    return last


def gsm8k_gold_final(gold: str) -> str:
    """
    GSM8K gold answers contain '#### <final>'.
    """
    if "####" in gold:
        tail = gold.split("####")[-1]
        return extract_last_number(tail)
    return extract_last_number(gold)


def gsm8k_match(pred: str, gold: str) -> bool:
    p = extract_last_number(pred)
    g = gsm8k_gold_final(gold)
    return (p != "") and (g != "") and (p == g)


def medmcqa_strict_match(pred: str, gold_letter: str) -> bool:
    """
    Strict: predicted contains one of {A,B,C,D} and equals the gold letter (after normalization).
    """
    g = gold_letter.strip().upper()
    if g not in ["A", "B", "C", "D"]:
        return False
    # try to extract a single letter answer from pred
    m = re.search(r"\b([ABCD])\b", pred.strip().upper())
    if not m:
        return False
    return m.group(1) == g


def medmcqa_relaxed_match(pred: str, gold_letter: str, prompt: str) -> bool:
    """
    Relaxed: accept strict letter match OR if the predicted text contains the correct option text.
    """
    if medmcqa_strict_match(pred, gold_letter):
        return True

    # parse option text from prompt
    # prompt contains lines "A) ...", etc.
    options = {}
    for line in prompt.splitlines():
        m = re.match(r"^\s*([ABCD])\)\s*(.*)$", line.strip())
        if m:
            options[m.group(1)] = m.group(2).strip()

    g = gold_letter.strip().upper()
    opt_text = options.get(g, "")
    if not opt_text:
        return False

    return relaxed_match(pred, opt_text)


# -------------------------
# LM utilities
# -------------------------

@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


@torch.no_grad()
def get_hidden_states_over_input(model, tokenizer, prompt: str, device: str) -> List[List[torch.Tensor]]:
    """
    Returns hidden states for each layer over the INPUT tokens only.
    Format matches your cls.py expectation:
      hs[layer] = list of token vectors, each [1, hidden_dim]
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # tuple length = num_layers+1, each [1, seq, dim]

    hs_layers: List[List[torch.Tensor]] = []
    for h in hidden_states:
        # h: [1, seq_len, dim]
        tokens = [h[:, i, :].detach().cpu() for i in range(h.shape[1])]  # each [1, dim]
        hs_layers.append(tokens)
    return hs_layers


# -------------------------
# Balancing + processing
# -------------------------

@dataclass
class ExampleOut:
    hs_by_layer: Dict[int, List[torch.Tensor]]
    label_strict: int
    label_relaxed: int
    prompt: str
    gold: str
    pred: str


def compute_labels(dataset: str, prompt: str, pred: str, gold: str) -> Tuple[int, int]:
    """
    Returns (label_strict, label_relaxed)
    strict: paper-style (EM / option-letter / numeric GSM8K)
    relaxed: more permissive string-based match (still deterministic)
    """
    if dataset == "gsm8k":
        strict = 1 if gsm8k_match(pred, gold) else 0
        # relaxed: accept numeric OR relaxed text (rarely useful, but deterministic)
        relaxed = 1 if (gsm8k_match(pred, gold) or relaxed_match(pred, gsm8k_gold_final(gold))) else 0
        return strict, relaxed

    if dataset == "medmcqa":
        # gold is usually a letter in our loader
        strict = 1 if medmcqa_strict_match(pred, gold) else 0
        relaxed = 1 if medmcqa_relaxed_match(pred, gold, prompt) else 0
        return strict, relaxed

    # nq_open default
    strict = 1 if exact_match(pred, gold) else 0
    relaxed = 1 if relaxed_match(pred, gold) else 0
    return strict, relaxed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=["nq_open", "medmcqa", "gsm8k"])
    parser.add_argument("--data-root", type=str, default="datasets")

    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--n-test", type=int, default=400)

    parser.add_argument("--balanced", action="store_true",
                        help="Collect roughly equal positives/negatives by subsampling. "
                             "If set, n-train/n-val/n-test are PER-CLASS targets.")
    parser.add_argument("--layers", default="0,last")
    parser.add_argument("--k-shot", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    # resolve device
    def resolve_device(req: str) -> str:
        if req == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return req

    args.device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to("cpu")
    elif args.device == "mps":
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, device_map="auto")

    loaders = {
        "nq_open": load_nq,
        "medmcqa": load_medmcqa,
        "gsm8k": load_gsm8k,
    }
    load_fn = loaders[args.dataset]

    # load splits (NOTE: your old code used nq_open/gsm8k val from train for convenience)
    train_raw = load_fn("train", max(args.n_train * 5, args.n_train) if args.balanced else args.n_train)
    if args.dataset in ("gsm8k", "nq_open"):
        val_raw = load_fn("train", max(args.n_val * 5, args.n_val) if args.balanced else args.n_val)
    else:
        val_raw = load_fn("validation", max(args.n_val * 5, args.n_val) if args.balanced else args.n_val)
    # Some datasets (e.g., nq_open) don't expose a "test" split on HF; fall back to validation.
    test_split = "test"
    if args.dataset == "nq_open":
        test_split = "validation"
    test_raw = load_fn(test_split, max(args.n_test * 5, args.n_test) if args.balanced else args.n_test)

    # model info (hidden_size, num_layers)
    sample = get_hidden_states_over_input(model, tokenizer, train_raw[0][0], args.device)
    num_layers = len(sample) - 1
    hidden_size = sample[0][0].shape[-1]
    print("\nModel info:")
    print(f"hidden_size={hidden_size}")
    print(f"num_layers={num_layers}")

    # parse layers
    if args.layers.strip().lower() == "all":
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
        return f"{prompt} {gold}".strip()

    def build_kshot_prompt(cur_prompt: str, cur_idx: int, train_pool: List[Tuple[str, str]], is_train: bool) -> str:
        if args.k_shot <= 0:
            return cur_prompt
        k = args.k_shot
        indices = list(range(len(train_pool)))
        if is_train and 0 <= cur_idx < len(indices):
            indices.pop(cur_idx)
        if k > len(indices):
            k = len(indices)
        shot_idx = rng.sample(indices, k)
        shots = [format_example(train_pool[i][0], train_pool[i][1]) for i in shot_idx]
        return "\n\n".join(shots + [cur_prompt])

    def collect_split(raw_split: List[Tuple[str, str]], split_name: str, is_train: bool, n_target: int):
        """
        If not balanced: collect first n_target examples.
        If balanced: collect n_target positives and n_target negatives (per-class).
        """
        hs_layers: Dict[int, List[List[torch.Tensor]]] = {l: [] for l in layers_to_save}
        labels_strict: List[int] = []
        labels_relaxed: List[int] = []
        prompts: List[str] = []
        golds: List[str] = []
        preds: List[str] = []
        meta: List[dict] = []

        if not args.balanced:
            iterator = enumerate(raw_split[:n_target])
            for idx, (prompt, gold) in tqdm(list(iterator), desc=split_name):
                full_prompt = build_kshot_prompt(prompt, idx, train_raw, is_train)
                pred = generate_answer(model, tokenizer, full_prompt, args.device, args.max_new_tokens)
                ls, lr = compute_labels(args.dataset, full_prompt, pred, gold)
                hs = get_hidden_states_over_input(model, tokenizer, full_prompt, args.device)

                for l in layers_to_save:
                    hs_layers[l].append(hs[l])

                labels_strict.append(ls)
                labels_relaxed.append(lr)
                prompts.append(full_prompt)
                golds.append(gold)
                preds.append(pred)
                meta.append({"prompt": full_prompt, "gold": gold, "pred": pred, "label_strict": ls, "label_relaxed": lr})
            return hs_layers, labels_strict, labels_relaxed, prompts, golds, preds, meta

        # balanced collection
        pos_needed = n_target
        neg_needed = n_target
        pos_count = 0
        neg_count = 0

        for idx, (prompt, gold) in tqdm(list(enumerate(raw_split)), desc=f"{split_name}(balanced)"):
            if pos_count >= pos_needed and neg_count >= neg_needed:
                break

            full_prompt = build_kshot_prompt(prompt, idx, train_raw, is_train)
            pred = generate_answer(model, tokenizer, full_prompt, args.device, args.max_new_tokens)
            ls, lr = compute_labels(args.dataset, full_prompt, pred, gold)

            # IMPORTANT: balancing uses STRICT labels (paper-aligned)
            label_for_balance = ls

            if label_for_balance == 1 and pos_count >= pos_needed:
                continue
            if label_for_balance == 0 and neg_count >= neg_needed:
                continue

            hs = get_hidden_states_over_input(model, tokenizer, full_prompt, args.device)

            for l in layers_to_save:
                hs_layers[l].append(hs[l])

            labels_strict.append(ls)
            labels_relaxed.append(lr)
            prompts.append(full_prompt)
            golds.append(gold)
            preds.append(pred)
            meta.append({"prompt": full_prompt, "gold": gold, "pred": pred, "label_strict": ls, "label_relaxed": lr})

            if label_for_balance == 1:
                pos_count += 1
            else:
                neg_count += 1

        print(f"[{split_name}] collected pos={pos_count}/{pos_needed}, neg={neg_count}/{neg_needed}, total={len(labels_strict)}")
        return hs_layers, labels_strict, labels_relaxed, prompts, golds, preds, meta

    model_dir = args.model.split("/")[-1]
    base = os.path.join(args.data_root, args.dataset, model_dir, "hs")
    os.makedirs(base, exist_ok=True)

    for split_name, raw, n_target, is_train in [
        ("train", train_raw, args.n_train, True),
        ("val",   val_raw,   args.n_val,   False),
        ("test",  test_raw,  args.n_test,  False),
    ]:
        out_dir = os.path.join(base, split_name)
        os.makedirs(out_dir, exist_ok=True)

        hs_layers, ls, lr, prompts, golds, preds, meta = collect_split(raw, split_name, is_train, n_target)

        for l in layers_to_save:
            torch.save(
                {
                    "hs": hs_layers[l],
                    "labels_strict": ls,
                    "labels_relaxed": lr,
                    "labels": ls,  # backward compatible default = strict
                    "prompts": prompts,
                    "golds": golds,
                    "preds": preds,
                    "meta": meta,
                },
                os.path.join(out_dir, f"layer_{l}.pth")
            )

    print("\nDone. Saved per-layer .pth with hs + labels_strict/labels_relaxed + preds/golds/prompts/meta.")


if __name__ == "__main__":
    main()
