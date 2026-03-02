#!/usr/bin/env python3
"""
Inference-time intervention for small HF causal LMs (GPT-2 family).

- Loads classifier f_theta (SimpleMLP) trained by cls.py
- Loads intervention model g_phi (3-layer MLP) trained separately for this LM dim
- If classifier prob(factual) <= alpha, intervene at layer l on the prompt pass:
    h_last <- h_last + tr * g_phi(h_last)

Outputs:
- base generation
- intervened generation
- classifier prob on prompt

Works best with: distilgpt2, gpt2, phi-2, qwen
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def get_hidden_size(model) -> int:
    cfg = model.config
    for attr in ("n_embd", "hidden_size", "n_hidden"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise AttributeError("Could not infer hidden size from model config.")


def get_block_list(model):
    """
    Return the list-like module of transformer blocks across common architectures.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h  # GPT-2 style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers   # LLaMA/Phi/Qwen style
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers  # GPT-NeoX style
    if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
        return model.transformer.layers
    raise AttributeError("Unsupported model architecture: cannot locate transformer blocks.")


def pool_hidden(hs_tokens, mode: int) -> torch.Tensor:
    """
    hs_tokens: list of [1, d] tensors
    mode: 0 mean, 1 max, else last
    returns [d]
    """
    x = torch.cat(hs_tokens, dim=0).float()  # [T, d]
    if mode == 0:
        return x.mean(dim=0)
    if mode == 1:
        return x.max(dim=0).values
    return x[-1]


class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 1024):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu(self.fc(x))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out


class InterventionMLP(nn.Module):
    """
    Small deterministic g_phi for GPT-2 dims.
    """
    def __init__(self, d: int, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        h1 = d * hidden_mult
        h2 = d * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, d),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def get_layer_tokens(model, input_ids, attention_mask, layer_idx: int):
    """
    Returns list of token vectors [1, d] for the given layer on prompt-only forward.
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hs_layer = out.hidden_states[layer_idx][0]  # [T, d]
    return [hs_layer[i].detach().cpu().unsqueeze(0) for i in range(hs_layer.shape[0])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilgpt2")
    p.add_argument("--prompt", required=True)

    # classifier
    p.add_argument("--cls-ckpt", required=True, help="path to classifier .pth (state_dict)")
    p.add_argument("--layer", type=int, required=True, help="layer index used by classifier & intervention")
    p.add_argument("--mode", type=int, default=0, help="0 mean, 1 max, else last")

    # intervention
    p.add_argument("--gphi-ckpt", required=True, help="path to g_phi .pth (state_dict)")
    p.add_argument("--tr", type=float, default=0.3, help="scaling for delta added to hidden state")
    p.add_argument("--alpha", type=float, default=0.3, help="intervene if prob(factual) <= alpha")

    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = resolve_device(args.device)

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    # Determine hidden size
    d = get_hidden_size(model)

    # Load classifier
    cls = SimpleMLP(input_size=d, hidden_size=1024).to(device)
    cls.load_state_dict(torch.load(args.cls_ckpt, map_location=device))
    cls.eval()

    # Load intervention model
    gphi = InterventionMLP(d=d, hidden_mult=2, dropout=0.1).to(device)
    gphi.load_state_dict(torch.load(args.gphi_ckpt, map_location=device))
    gphi.eval()

    enc = tok(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    # classifier prob on prompt-only hidden states at layer args.layer
    hs_tokens = get_layer_tokens(model, input_ids, attention_mask, args.layer)
    pooled = pool_hidden(hs_tokens, args.mode).to(device)  # [d]
    prob_factual = float(cls(pooled.unsqueeze(0)).squeeze().item())
    do_intervene = prob_factual <= args.alpha

    # baseline generation
    base_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    base_text = tok.decode(base_out[0], skip_special_tokens=True)

    # intervened generation via forward hook on the specified block
    applied = {"flag": False}

    def hook_fn(module, inputs, output):
        # GPT-2 block output is hidden_states tensor [B, T, d]
        if applied["flag"]:
            return output
        hidden = output
        if not isinstance(hidden, torch.Tensor):
            return output

        # Apply only on the prompt pass: seq_len == prompt_len
        if hidden.shape[1] != prompt_len:
            return output

        if not do_intervene:
            applied["flag"] = True
            return output

        # last token hidden state at this layer
        last = hidden[:, -1, :]  # [B, d]
        delta = gphi(last)       # [B, d]
        hidden = hidden.clone()
        hidden[:, -1, :] = last + (args.tr * delta)

        applied["flag"] = True
        return hidden

    # Register hook on the specified block
    blocks = get_block_list(model)
    if args.layer > 0:
        if args.layer - 1 >= len(blocks):
            raise ValueError(f"layer={args.layer} out of range for model blocks (len={len(blocks)})")
        handle = blocks[args.layer - 1].register_forward_hook(hook_fn)
    else:
        handle = None
    # Note: layer 0 is embeddings output; for intervention you usually want >=1.

    int_out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    if handle is not None:
        handle.remove()

    int_text = tok.decode(int_out[0], skip_special_tokens=True)

    print("=== Prompt ===")
    print(args.prompt)
    print("\n=== Classifier ===")
    print(f"layer={args.layer} mode={args.mode} prob_factual={prob_factual:.4f} alpha={args.alpha} intervene={do_intervene}")
    print("\n=== Base ===")
    print(base_text)
    print("\n=== Intervened ===")
    print(int_text)


if __name__ == "__main__":
    main()
