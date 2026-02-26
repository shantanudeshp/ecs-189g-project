import argparse
import json
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

try:
    import wandb
except Exception:  # pragma: no cover - fallback when wandb isn't available
    class _DummyWandb:
        def init(self, *args, **kwargs):
            return None

        def log(self, *args, **kwargs):
            return None

    wandb = _DummyWandb()


# ---------------------------
# Reproducibility
# ---------------------------
DEFAULT_SEED = 42
torch.manual_seed(DEFAULT_SEED)
np.random.seed(DEFAULT_SEED)
random.seed(DEFAULT_SEED)


def resolve_device(requested: str = "auto") -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


# ---------------------------
# Model metadata (extend as needed)
# ---------------------------
MODEL_LAYER_EMB_MAP = {
    "distilgpt2": {"num_layers": 6, "emb_size": 768},
    "gpt2": {"num_layers": 12, "emb_size": 768},
}

MODELS_IDS = {
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2",
}


# ---------------------------
# Pooling over token vectors
# ---------------------------
def _ensure_token_matrix(hs: List[torch.Tensor]) -> torch.Tensor:
    """
    hs: list of tensors, usually one per token.
        common shapes:
          - [1, hidden]
          - [hidden]
    returns: [T, hidden] float32
    """
    if len(hs) == 0:
        raise ValueError("Empty hidden-state list encountered.")

    rows = []
    for t in hs:
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t)
        t = t.detach().cpu()
        if t.ndim == 2 and t.shape[0] == 1:
            t = t.squeeze(0)  # [hidden]
        elif t.ndim == 1:
            pass
        else:
            # If load_classifier ever stores something like [seq, hidden] per item, handle it.
            # Flatten to token rows if needed.
            if t.ndim == 2:
                # treat as already [seq, hidden] and append all rows
                for r in t:
                    rows.append(r)
                continue
            raise ValueError(f"Unexpected hidden state tensor shape: {tuple(t.shape)}")

        rows.append(t)

    mat = torch.stack(rows, dim=0).float()  # [T, hidden]
    return mat


def pool_hidden_states(hs: List[torch.Tensor], mode: int, emb_size: int) -> torch.Tensor:
    """
    mode:
      0 -> mean pooling across tokens
      1 -> max pooling across tokens
     -1 -> last token vector
    returns: [hidden]
    """
    mat = _ensure_token_matrix(hs)  # [T, hidden]
    if mat.shape[-1] != emb_size:
        raise ValueError(f"Hidden size mismatch: got {mat.shape[-1]}, expected {emb_size}")

    if mode == 0:
        return mat.mean(dim=0)
    if mode == 1:
        return mat.max(dim=0).values
    # default / -1
    return mat[-1]


# ---------------------------
# Simple MLP probe
# ---------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # binary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.fc(x))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out


class CustomDataset(Dataset):
    def __init__(self, data: List[torch.Tensor], labels: List[int]):
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.data[idx], self.labels[idx]


def collate_fn(batch):
    data, labels = zip(*batch)
    data_tensor = torch.stack(data) if all(isinstance(d, torch.Tensor) for d in data) else torch.tensor(data)
    labels_tensor = torch.tensor(labels)
    return data_tensor, labels_tensor


# ---------------------------
# Metrics
# ---------------------------
def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    rounded_preds = torch.round(outputs)
    correct = (rounded_preds == labels).float()
    return correct.sum() / len(correct)


def f1(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = torch.round(outputs.detach())
    return f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average="macro", zero_division=0)


def precision(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    rounded_preds = torch.round(outputs)
    tp = ((rounded_preds == 1) & (labels == 1)).float().sum()
    pp = (rounded_preds == 1).float().sum()
    return (tp / pp).item() if pp != 0 else 0.0


def recall(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    rounded_preds = torch.round(outputs)
    tp = ((rounded_preds == 1) & (labels == 1)).float().sum()
    ap = (labels == 1).float().sum()
    return (tp / ap).item() if ap != 0 else 0.0


def false_positives(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    rounded_preds = torch.round(outputs)
    fp = ((rounded_preds == 1) & (labels == 0)).float().sum()
    return fp.item()


def false_negatives(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    rounded_preds = torch.round(outputs)
    fn = ((rounded_preds == 0) & (labels == 1)).float().sum()
    return fn.item()


# ---------------------------
# Train / Eval loops
# ---------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    criterion,
    optimizer,
    best_model_path: str,
    device: str,
    use_wandb: bool = True,
    log_json_path: str | None = None,
):
    best_val_accuracy = 0.0
    model.to(device)

    metrics = []
    for epoch in range(epochs):
        model.train()
        train_loss = train_acc = train_f1 = 0.0
        train_prec = train_rec = 0.0
        train_fp = train_fn = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float()).squeeze(-1)
            loss = criterion(outputs, labels.float())

            train_loss += loss.item()
            train_acc += accuracy(outputs, labels.float()).item()
            train_f1 += f1(outputs, labels.float())
            train_prec += precision(outputs, labels.float())
            train_rec += recall(outputs, labels.float())
            train_fp += false_positives(outputs, labels.float())
            train_fn += false_negatives(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        n_batches = max(1, len(train_loader))
        train_loss /= n_batches
        train_acc /= n_batches
        train_f1 /= n_batches
        train_prec /= n_batches
        train_rec /= n_batches
        train_fp /= n_batches
        train_fn /= n_batches

        print(
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
            f"Train Precision: {train_prec:.4f}, Train Recall: {train_rec:.4f}, "
            f"Train False Positives: {train_fp:.4f}, Train False Negatives: {train_fn:.4f}"
        )

        model.eval()
        val_loss = val_acc = val_f1 = 0.0
        val_prec = val_rec = 0.0
        val_fp = val_fn = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float()).squeeze(-1)
                loss = criterion(outputs, labels.float())

                val_loss += loss.item()
                val_acc += accuracy(outputs, labels.float()).item()
                val_f1 += f1(outputs, labels.float())
                val_prec += precision(outputs, labels.float())
                val_rec += recall(outputs, labels.float())
                val_fp += false_positives(outputs, labels.float())
                val_fn += false_negatives(outputs, labels.float())

        n_batches = max(1, len(val_loader))
        val_loss /= n_batches
        val_acc /= n_batches
        val_f1 /= n_batches
        val_prec /= n_batches
        val_rec /= n_batches
        val_fp /= n_batches
        val_fn /= n_batches

        print(
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}, "
            f"Val Precision: {val_prec:.4f}, Val Recall: {val_rec:.4f}, "
            f"Val False Positives: {val_fp:.4f}, Val False Negatives: {val_fn:.4f}"
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "train_f1": float(train_f1),
            "train_precision": float(train_prec),
            "train_recall": float(train_rec),
            "train_false_positives": float(train_fp),
            "train_false_negatives": float(train_fn),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_f1": float(val_f1),
            "val_precision": float(val_prec),
            "val_recall": float(val_rec),
            "val_false_positives": float(val_fp),
            "val_false_negatives": float(val_fn),
        }
        metrics.append(epoch_metrics)

        if use_wandb:
            wandb.log(epoch_metrics)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"** New best model saved at Epoch {epoch} with Val Accuracy: {val_acc:.4f} **")

    if log_json_path:
        os.makedirs(os.path.dirname(log_json_path), exist_ok=True)
        with open(log_json_path, "w") as f:
            json.dump({"epochs": metrics}, f, indent=2)

    return metrics


def test_model(model: nn.Module, test_loader: DataLoader, criterion, device: str, use_wandb: bool = True):
    model.to(device)
    model.eval()

    test_loss = test_acc = test_f1 = 0.0
    test_prec = test_rec = 0.0
    test_fp = test_fn = 0.0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float()).squeeze(-1)

            test_loss += criterion(outputs, labels.float()).item()
            test_acc += accuracy(outputs, labels.float()).item()
            test_f1 += f1(outputs, labels.float())
            test_prec += precision(outputs, labels.float())
            test_rec += recall(outputs, labels.float())
            test_fp += false_positives(outputs, labels.float())
            test_fn += false_negatives(outputs, labels.float())

    n_batches = max(1, len(test_loader))
    test_loss /= n_batches
    test_acc /= n_batches
    test_f1 /= n_batches
    test_prec /= n_batches
    test_rec /= n_batches
    test_fp /= n_batches
    test_fn /= n_batches

    test_metrics = {
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_f1": float(test_f1),
        "test_precision": float(test_prec),
        "test_recall": float(test_rec),
        "test_false_positives": float(test_fp),
        "test_false_negatives": float(test_fn),
    }

    if use_wandb:
        wandb.log(test_metrics)

    print(
        f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, "
        f"Test Precision: {test_prec:.4f}, Test Recall: {test_rec:.4f}, "
        f"Test False Positives: {test_fp:.4f}, Test False Negatives: {test_fn:.4f}"
    )
    return test_metrics


# ---------------------------
# IO helpers
# ---------------------------
def _load_split(data_root: str, dataset_name: str, model_id: str, hs_dir: str, split: str, layer: int) -> Dict[str, Any]:
    path = os.path.join(data_root, dataset_name, model_id, hs_dir, split, f"layer_{layer}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    # weights_only=False for torch>=2.0 compatibility with dict payloads
    return torch.load(path, map_location=torch.device("cpu"), weights_only=False)


def _get_labels(payload: Dict[str, Any], label_key: str) -> List[int]:
    if label_key not in payload:
        keys = sorted(list(payload.keys()))
        raise KeyError(f"Label key '{label_key}' not found in payload. Available keys: {keys}")
    labels = payload[label_key]
    # normalize to python ints
    return [int(x) for x in labels]


def _balance_pairs(pairs: List[Tuple[torch.Tensor, int]], seed: int) -> List[Tuple[torch.Tensor, int]]:
    rng = random.Random(seed)
    pos = [p for p in pairs if p[1] == 1]
    neg = [p for p in pairs if p[1] == 0]
    if len(pos) == 0 or len(neg) == 0:
        print("Warning: balance requested but one class is empty; skipping balancing.")
        return pairs
    k = min(len(pos), len(neg))
    pos = rng.sample(pos, k)
    neg = rng.sample(neg, k)
    out = pos + neg
    rng.shuffle(out)
    return out


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Train classifier on hidden states.")
    parser.add_argument("model_id", type=str, help="Enter the model ID (alias or HF id)")
    parser.add_argument("dataset_name", type=str, help="Enter the dataset name")
    parser.add_argument(
        "modes",
        nargs="*",
        type=int,
        default=[0],
        help="Pooling modes: 0=mean, 1=max, -1=last-token. Accepts multiple modes separated by space.",
    )

    parser.add_argument("--data-root", type=str, default="datasets", help="Root directory for datasets")
    parser.add_argument("--hs-dir", type=str, default="hs", help="Hidden-state directory name under dataset/model")
    parser.add_argument("--layers", type=str, default="all", help="Comma-separated layer list or 'all'")

    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=1024, help="MLP hidden size")

    parser.add_argument("--num-layers", type=int, default=None, help="Override number of layers")
    parser.add_argument("--emb-size", type=int, default=None, help="Override embedding size")

    parser.add_argument("--save-dir", type=str, default="clss", help="Output directory for classifier checkpoints")

    parser.add_argument("--wandb-project", type=str, default=None, help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--wandb-run-prefix", type=str, default="", help="Prefix for W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, or mps")
    parser.add_argument("--log-json", type=str, default=None, help="Write metrics to JSON at this path")

    parser.add_argument("--balance-train", action="store_true", help="Downsample majority class in training data")
    parser.add_argument("--balance-seed", type=int, default=42, help="Seed for train balancing")

    # NEW: choose which labels to train on
    parser.add_argument(
        "--label-key",
        type=str,
        default="labels",
        help="Which label field to read from the .pth payload (e.g., labels, labels_em, labels_num).",
    )

    args = parser.parse_args()
    device = resolve_device(args.device)

    # Resolve model meta
    model_id_in = args.model_id
    if model_id_in in MODEL_LAYER_EMB_MAP:
        emb_size = MODEL_LAYER_EMB_MAP[model_id_in]["emb_size"]
        num_layers = MODEL_LAYER_EMB_MAP[model_id_in]["num_layers"]
    else:
        if args.emb_size is None or args.num_layers is None:
            raise ValueError("Unknown model_id. Provide --emb-size and --num-layers.")
        emb_size = args.emb_size
        num_layers = args.num_layers

    model_id_path = MODELS_IDS.get(model_id_in, model_id_in)
    model_id = model_id_path.split("/")[-1]  # used in folder paths

    if args.layers.strip().lower() == "all":
        layers = list(range(num_layers + 1))
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]

    os.makedirs(f"{args.save_dir}/{args.dataset_name}/{model_id}", exist_ok=True)

    use_wandb = (not args.no_wandb) and (args.wandb_project is not None)

    for mode in args.modes:
        for layer in layers:
            train_payload = _load_split(args.data_root, args.dataset_name, model_id, args.hs_dir, "train", layer)
            val_payload = _load_split(args.data_root, args.dataset_name, model_id, args.hs_dir, "val", layer)
            test_payload = _load_split(args.data_root, args.dataset_name, model_id, args.hs_dir, "test", layer)

            train_labels = _get_labels(train_payload, args.label_key)
            val_labels = _get_labels(val_payload, args.label_key)
            test_labels = _get_labels(test_payload, args.label_key)

            # Build (pooled_vector, label) pairs
            train_pairs = [(pool_hidden_states(hs, mode, emb_size), y) for hs, y in zip(train_payload["hs"], train_labels)]
            val_pairs = [(pool_hidden_states(hs, mode, emb_size), y) for hs, y in zip(val_payload["hs"], val_labels)]
            test_pairs = [(pool_hidden_states(hs, mode, emb_size), y) for hs, y in zip(test_payload["hs"], test_labels)]

            if args.balance_train:
                train_pairs = _balance_pairs(train_pairs, args.balance_seed)

            print(f"[layer={layer} mode={mode} label_key={args.label_key}] sizes:",
                  f"train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

            train_dataset = CustomDataset([p[0] for p in train_pairs], [p[1] for p in train_pairs])
            val_dataset = CustomDataset([p[0] for p in val_pairs], [p[1] for p in val_pairs])
            test_dataset = CustomDataset([p[0] for p in test_pairs], [p[1] for p in test_pairs])

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

            run_name = f"{args.wandb_run_prefix}label_{args.label_key}__m_{mode}_l_{layer}_b_{args.batch_size}"
            if use_wandb:
                wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, reinit=True)

            best_model_path = f"{args.save_dir}/{args.dataset_name}/{model_id}/{run_name}.pth"

            model = SimpleMLP(emb_size, args.hidden_size, 1)
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=args.lr)

            log_json_path = args.log_json
            if log_json_path:
                log_json_path = (
                    log_json_path.replace("{dataset}", args.dataset_name)
                    .replace("{model}", model_id)
                    .replace("{layer}", str(layer))
                    .replace("{mode}", str(mode))
                    .replace("{label_key}", args.label_key)
                )

            train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                criterion=criterion,
                optimizer=optimizer,
                best_model_path=best_model_path,
                device=device,
                use_wandb=use_wandb,
                log_json_path=log_json_path,
            )

            model.load_state_dict(torch.load(best_model_path, weights_only=True))
            test_metrics = test_model(model, test_loader, criterion, device=device, use_wandb=use_wandb)

            if log_json_path:
                # append test metrics
                with open(log_json_path, "r+") as f:
                    payload = json.load(f)
                    payload["test"] = test_metrics
                    f.seek(0)
                    json.dump(payload, f, indent=2)
                    f.truncate()


if __name__ == "__main__":
    main()