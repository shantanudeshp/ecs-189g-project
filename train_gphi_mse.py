#!/usr/bin/env python3
"""
train_gphi_mse.py

Train a small deterministic intervention model g_phi for a small HF causal LM
(e.g., distilgpt2/gpt2). This is Step D.

Training data format (produced by build_intervention_dataset.py):
  datasets/<dataset>/<model_name>/hs/train_fct/layer_<L>.pth
    {
      "hs":     List[List[Tensor]]     # token vectors for (prompt + model_answer)
      "hs_fct": List[List[Tensor]]     # token vectors for target sequence:
                                       #   if correct -> same as hs
                                       #   else -> (prompt + gold_answer)
      "meta":   ... (optional)
    }

We train g_phi: R^d -> R^d such that:
  h_last + tr * g_phi(h_last)  ~=  h_target_last

Loss: MSE on last-token vectors.

Outputs:
  gphi_ckpts/<dataset>/<model_name>/layer_<L>/gphi_mse.pth

Usage example:
  python train_gphi_mse.py --model distilgpt2 --dataset nq_open --layer 6 --device cuda
"""

import argparse
import os
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def last_token_vec(token_list: List[torch.Tensor]) -> torch.Tensor:
    """
    token_list: list of [1, d] tensors
    returns: [d] tensor
    """
    return token_list[-1].squeeze(0).float()


class PairDataset(Dataset):
    def __init__(self, hs: List[List[torch.Tensor]], hs_fct: List[List[torch.Tensor]]):
        assert len(hs) == len(hs_fct)
        self.x = [last_token_vec(h) for h in hs]
        self.y = [last_token_vec(h) for h in hs_fct]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


class GphiMLP(nn.Module):
    """
    Deterministic 3-layer MLP: d -> (2d) -> (2d) -> d
    """
    def __init__(self, d: int, hidden_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        h = d * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_fct_layer(data_root: str, dataset: str, model_name: str, split: str, layer: int):
    path = os.path.join(data_root, dataset, model_name, "hs", split, f"layer_{layer}.pth")
    obj = torch.load(path, map_location="cpu")
    if "hs" not in obj or "hs_fct" not in obj:
        raise ValueError(f"{path} missing required keys. Found keys={list(obj.keys())}")
    return obj["hs"], obj["hs_fct"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="distilgpt2", help="HF model id used to build datasets")
    p.add_argument("--dataset", required=True, choices=["nq_open", "medmcqa", "gsm8k", "ai2_arc"])
    p.add_argument("--layer", type=int, required=True, help="layer index used in train_fct/layer_<L>.pth")
    p.add_argument("--data-root", default="datasets")

    p.add_argument("--tr", type=float, default=0.3, help="scaling used in training: h + tr*g(h)")
    p.add_argument("--hidden-mult", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    model_name = args.model.split("/")[-1]

    # Load train/val
    train_hs, train_hs_fct = load_fct_layer(args.data_root, args.dataset, model_name, "train_fct", args.layer)
    val_hs, val_hs_fct = load_fct_layer(args.data_root, args.dataset, model_name, "val_fct", args.layer)

    train_ds = PairDataset(train_hs, train_hs_fct)
    val_ds = PairDataset(val_hs, val_hs_fct)

    d = train_ds[0][0].shape[0]
    print(f"Loaded intervention dataset: train={len(train_ds)} val={len(val_ds)} hidden_size={d}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    gphi = GphiMLP(d=d, hidden_mult=args.hidden_mult, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(gphi.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse = nn.MSELoss()

    best_val = float("inf")
    out_dir = os.path.join("gphi_ckpts", args.dataset, model_name, f"layer_{args.layer}")
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, "gphi_mse.pth")

    for epoch in range(1, args.epochs + 1):
        gphi.train()
        total_loss = 0.0
        n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            delta = gphi(x)
            x_adj = x + args.tr * delta
            loss = mse(x_adj, y)
            loss.backward()
            opt.step()

            bs = x.shape[0]
            total_loss += float(loss.item()) * bs
            n += bs

        train_loss = total_loss / max(n, 1)

        # Validation
        gphi.eval()
        vloss = 0.0
        vn = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                delta = gphi(x)
                x_adj = x + args.tr * delta
                loss = mse(x_adj, y)
                bs = x.shape[0]
                vloss += float(loss.item()) * bs
                vn += bs

        val_loss = vloss / max(vn, 1)
        print(f"Epoch {epoch:02d}/{args.epochs}  train_mse={train_loss:.6f}  val_mse={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(gphi.state_dict(), ckpt_path)

    print(f"Saved best gphi checkpoint to: {ckpt_path}")
    print("Use it in intervene_decode.py with --gphi-ckpt")


if __name__ == "__main__":
    main()
