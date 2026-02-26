#!/usr/bin/env python3
"""
Judge harness for win/tie/loss.

Input JSONL lines:
  {"question": "...", "first": "...", "second": "..."}

Decision labels:
  "first" | "second" | "both" | "neither"

By default uses Ollama local endpoint.
"""

import argparse
import json
import re
import requests
from typing import Dict, Any


JUDGE_SYSTEM = (
    "You are an expert evaluator. Compare two responses to a question for factual accuracy. "
    "Ignore style and length unless they change factual meaning. Decide which response is more factually correct."
)

JUDGE_USER_TMPL = """Question:
{question}

Response 1:
{first}

Response 2:
{second}

Output exactly one word from: first, second, both, neither.
"""


def call_ollama(model: str, prompt: str, system: str, url: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def parse_decision(text: str) -> str:
    t = text.strip().lower()
    m = re.search(r"\b(first|second|both|neither)\b", t)
    return m.group(1) if m else "neither"


def to_wtl(decision: str) -> str:
    # win = second, loss = first, tie = both/neither
    if decision == "second":
        return "win"
    if decision == "first":
        return "loss"
    return "tie"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-jsonl", required=True)
    p.add_argument("--out-jsonl", required=True)

    # ollama
    p.add_argument("--backend", choices=["ollama"], default="ollama")
    p.add_argument("--ollama-model", default="qwen2.5:7b-instruct")
    p.add_argument("--ollama-url", default="http://localhost:11434/api/chat")

    args = p.parse_args()

    win = tie = loss = 0

    with open(args.in_jsonl, "r", encoding="utf-8") as f_in, open(args.out_jsonl, "w", encoding="utf-8") as f_out:
        for line in f_in:
            ex: Dict[str, Any] = json.loads(line)
            q = ex["question"]
            first = ex["first"]
            second = ex["second"]

            user_prompt = JUDGE_USER_TMPL.format(question=q, first=first, second=second)
            raw = call_ollama(args.ollama_model, user_prompt, JUDGE_SYSTEM, args.ollama_url)
            decision = parse_decision(raw)
            wtl = to_wtl(decision)

            if wtl == "win":
                win += 1
            elif wtl == "loss":
                loss += 1
            else:
                tie += 1

            out = {
                "question": q,
                "first": first,
                "second": second,
                "judge_raw": raw,
                "decision": decision,
                "wtl": wtl,
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")

    total = win + tie + loss
    print(f"Total={total}  win={win}  tie={tie}  loss={loss}")


if __name__ == "__main__":
    main()