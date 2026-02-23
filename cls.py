import torch
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score  
import torch.optim as optim
try:
    import wandb
except Exception:  # pragma: no cover - fallback when wandb isn't available
    class _DummyWandb:
        def init(self, *args, **kwargs):
            return None
        def log(self, *args, **kwargs):
            return None
    wandb = _DummyWandb()
import argparse
import os

# Set a seed value
seed = 42  # You can choose any seed number

# Set the random seed for various libraries
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

def resolve_device(requested: str = "auto") -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


# MODES = [0, -1, 1]
MODEL_LAYER_EMB_MAP = {
    "distilgpt2": {
    "num_layers": 6,
    "emb_size": 768
    },
    "gpt2": {
        "num_layers": 12,
        "emb_size": 768
    }
}

MODELS_IDS = {
    "distilgpt2": "distilgpt2",
    "gpt2": "gpt2"
}

def map_selected_mode(hs, mode):
    # Convert list of tensors into a single tensor
    # Each tensor is of shape [1, 4096], we concatenate along dim=0
    tensor = torch.cat(hs, dim=0).float()  # Resulting tensor shape will be [len(hs), 1, 4096]

    # Squeeze the middle dimension to make the shape [len(hs), 4096]
    tensor = tensor.squeeze(1)  # Now tensor shape is [len(hs), 4096]

    if mode == 0:
        # Mean pooling across the new concatenated dimension
        mean_pooled = torch.mean(tensor, dim=0)
        # print("mean_pooled.shape[-1]", mean_pooled.shape[-1])
        assert mean_pooled.shape[-1] == emb_size #5120
        return mean_pooled
    
    elif mode == 1:
        # Max pooling across the new concatenated dimension
        max_pooled, _ = torch.max(tensor, dim=0)
        # print("max_pooled.shape[-1]", max_pooled.shape[-1])
        assert max_pooled.shape[-1] == emb_size #5120
        return max_pooled
    
    else:
        # Return the last sequence element of the last tensor if mode is neither 0 nor 1
        last = hs[-1].squeeze(0)
        assert last.shape[-1] == emb_size #5120
        return last  # Ensure it returns a tensor of shape [4096]
    

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden to Output Layer
        self.sigmoid = nn.Sigmoid()  # Since it's a binary classification

    def forward(self, x):
        out = self.relu(self.fc(x))
        out = self.dropout(out)
        out = self.sigmoid(self.fc2(out))
        return out
    
# (tensor_id, tensor, label)
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    
def collate_fn(batch):
    data, labels = zip(*batch)
    if all(isinstance(d, torch.Tensor) for d in data):
        data_tensor = torch.stack(data)
    else:
        data_tensor = torch.tensor(data)
    labels_tensor = torch.tensor(labels)
    return data_tensor, labels_tensor

def accuracy(outputs, labels):
    rounded_preds = torch.round(outputs)
    correct = (rounded_preds == labels).float()  # Convert into float for division
    acc = correct.sum() / len(correct)
    return acc

def f1(outputs, labels):
    predictions = torch.round(outputs.detach())
    f1_score_value = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro', zero_division=0)
    return f1_score_value

def precision(outputs, labels):
    rounded_preds = torch.round(outputs)
    true_positives = ((rounded_preds == 1) & (labels == 1)).float().sum()
    predicted_positives = (rounded_preds == 1).float().sum()
    precision = true_positives / predicted_positives if predicted_positives != 0 else 0
    return precision

def recall(outputs, labels):
    rounded_preds = torch.round(outputs)
    true_positives = ((rounded_preds == 1) & (labels == 1)).float().sum()
    actual_positives = (labels == 1).float().sum()
    recall = true_positives / actual_positives if actual_positives != 0 else 0
    return recall

def false_positives(outputs, labels):
    rounded_preds = torch.round(outputs)
    false_positives = ((rounded_preds == 1) & (labels == 0)).float().sum()
    return false_positives.item()

def false_negatives(outputs, labels):
    rounded_preds = torch.round(outputs)
    false_negatives = ((rounded_preds == 0) & (labels == 1)).float().sum()
    return false_negatives.item()


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, best_model_path, use_wandb=True):
    
    best_val_accuracy = 0.0
    best_val_loss = 1000
    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_acc, train_f1, train_precision, train_recall, train_false_positives, train_false_negatives = 0, 0, 0, 0, 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())
            loss = criterion(outputs.squeeze(-1), labels.float())

            train_loss += loss.item()
            train_acc += accuracy(outputs.squeeze(-1), labels.float())
            train_f1 += f1(outputs.squeeze(-1), labels.float())

            train_precision += precision(outputs.squeeze(-1), labels.float())
            train_recall += recall(outputs.squeeze(-1), labels.float())
            train_false_positives += false_positives(outputs.squeeze(-1), labels.float())
            train_false_negatives += false_negatives(outputs.squeeze(-1), labels.float())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_f1 /= len(train_loader)

        train_precision/= len(train_loader)
        train_recall/= len(train_loader)
        train_false_positives/= len(train_loader)
        train_false_negatives/= len(train_loader)
        
        # print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train F1: {train_f1}')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Train F1: {train_f1:.4f}, '
            f'Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, '
            f'Train False Positives: {train_false_positives:.4f}, Train False Negatives: {train_false_negatives:.4f}')

        # Validation phase
        model.eval()
        val_loss, val_acc, val_f1, val_precision, val_recall, val_false_positives, val_false_negatives = 0, 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs.float())
                loss = criterion(outputs.squeeze(-1), labels.float())
                val_loss += loss.item()
                val_acc += accuracy(outputs.squeeze(-1), labels.float())
                val_f1 += f1(outputs.squeeze(-1), labels.float())

                val_precision += precision(outputs.squeeze(-1), labels.float())
                val_recall += recall(outputs.squeeze(-1), labels.float())
                val_false_positives += false_positives(outputs.squeeze(-1), labels.float())
                val_false_negatives += false_negatives(outputs.squeeze(-1), labels.float())
                

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        val_f1 /= len(val_loader)

        val_precision/= len(val_loader)
        val_recall/= len(val_loader)
        val_false_positives/= len(val_loader)
        val_false_negatives/= len(val_loader)

        # print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Val F1: {val_f1:.4f}, '
            f'Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, '
            f'Val False Positives: {val_false_positives:.4f}, Val False Negatives: {val_false_negatives:.4f}')


        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_f1': train_f1,
                'train_precision': train_precision,
                'train_recall': train_recall,
                'train_false_positives': train_false_positives,
                'train_false_negatives': train_false_negatives,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_false_positives': val_false_positives,
                'val_false_negatives': val_false_negatives
            })

        # if val_loss < best_val_loss:
        if val_acc > best_val_accuracy:
            # best_val_loss = val_loss
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"** New best model saved at Epoch {epoch} with Val Accuracy: {val_acc} and Val Loss: {val_loss} **")

    return


def test_model(model, test_loader, criterion, use_wandb=True):
    model.to(device)
    model.eval()
    test_loss, test_acc, test_f1, test_precision, test_recall, test_false_positives, test_false_negatives = 0, 0, 0, 0, 0, 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs.float())

            test_loss += criterion(outputs.squeeze(-1), labels.float()).item()
            test_acc += accuracy(outputs.squeeze(-1), labels.float())
            test_f1 += f1(outputs.squeeze(-1), labels.float())
            
            test_precision += precision(outputs.squeeze(-1), labels.float())
            test_recall += recall(outputs.squeeze(-1), labels.float())
            test_false_positives += false_positives(outputs.squeeze(-1), labels.float())
            test_false_negatives += false_negatives(outputs.squeeze(-1), labels.float())


    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    test_f1 /= len(test_loader)

    test_precision /= len(test_loader)
    test_recall /= len(test_loader)
    test_false_positives /= len(test_loader)
    test_false_negatives /= len(test_loader)

    test_metrics = {
    'test_loss': test_loss,
    'test_acc': test_acc,
    'test_f1': test_f1,
    'test_precision': test_precision,
    'test_recall': test_recall,
    'test_false_positives': test_false_positives,
    'test_false_negatives': test_false_negatives
}
    if use_wandb:
        wandb.log(test_metrics)

    # print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1}')
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test F1: {test_f1:.4f}, '
      f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, '
      f'Test False Positives: {test_false_positives:.4f}, Test False Negatives: {test_false_negatives:.4f}')


parser = argparse.ArgumentParser(description="Train classifier on hidden states.")
parser.add_argument("model_id", type=str, help="Enter the model ID")
parser.add_argument("dataset_name", type=str, help="Enter the dataset name")
parser.add_argument("modes", nargs='*', type=int, default=[0],
                        help="Pass the modes, e.g., 0, 1, or -1. Accepts multiple modes separated by space.")
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
# [0, -1, 1] 

#modes [0, -1, 1]
# Parse arguments
args = parser.parse_args()
device = resolve_device(args.device)
model_id = args.model_id
dataset_name = args.dataset_name
modes = args.modes

if model_id in MODEL_LAYER_EMB_MAP:
    emb_size = MODEL_LAYER_EMB_MAP[model_id]["emb_size"]
    num_layers = MODEL_LAYER_EMB_MAP[model_id]["num_layers"]
else:
    if args.emb_size is None or args.num_layers is None:
        raise ValueError("Unknown model_id. Provide --emb-size and --num-layers.")
    emb_size = args.emb_size
    num_layers = args.num_layers

model_id_path = MODELS_IDS.get(args.model_id, args.model_id)
model_id = model_id_path.split("/")[-1] # update the model id to be part of the path

if args.layers.strip().lower() == "all":
    layers = list(range(num_layers + 1))
else:
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

# Check if the directory exists, and create it if it doesn't
os.makedirs(f"{args.save_dir}/{dataset_name}/{model_id}", exist_ok=True)

def _load_split(data_root, dataset_name, model_id, hs_dir, split, layer):
    path = os.path.join(data_root, dataset_name, model_id, hs_dir, split, f"layer_{layer}.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split file: {path}")
    return torch.load(path, map_location=torch.device('cpu'))

for mode in modes:
    for layer in layers:

        # train_path = f"datasets/{dataset_name}/{model_id}/hs/train/layer_{layer}.pth"
        # val_path = f"datasets/{dataset_name}/{model_id}/hs/val/layer_{layer}.pth"
        # test_path = f"datasets/{dataset_name}/{model_id}/hs/test/layer_{layer}.pth"
        
        train_data = _load_split(args.data_root, dataset_name, model_id, args.hs_dir, "train", layer)
        val_data = _load_split(args.data_root, dataset_name, model_id, args.hs_dir, "val", layer)
        test_data = _load_split(args.data_root, dataset_name, model_id, args.hs_dir, "test", layer)

        train_data = [(map_selected_mode(tensors, mode), label) for tensors, label in zip(train_data['hs'], train_data['labels'])]
        val_data = [(map_selected_mode(tensors, mode), label) for tensors, label in zip(val_data['hs'], val_data['labels'])]
        test_data = [(map_selected_mode(tensors, mode), label) for tensors, label in zip(test_data['hs'], test_data['labels'])]

        print("len train_data: ", len(train_data))
        print("len val_data: ", len(val_data))
        print("len test_data: ", len(test_data))

        # Creating datasets using the correct format
        train_dataset = CustomDataset([t[0] for t in train_data], [t[1] for t in train_data])
        val_dataset = CustomDataset([t[0] for t in val_data], [t[1] for t in val_data])
        test_dataset = CustomDataset([t[0] for t in test_data], [t[1] for t in test_data])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        run_name = f"{args.wandb_run_prefix}m_{mode}_l_{layer}_b_{args.batch_size}"
        use_wandb = not args.no_wandb and args.wandb_project is not None
        if use_wandb:
            wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=run_name, reinit=True)

        best_model_path = f"{args.save_dir}/{dataset_name}/{model_id}/{run_name}.pth"

        model = SimpleMLP(emb_size, args.hidden_size, 1)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        train_model(model, train_loader, val_loader, epochs=args.epochs, criterion=criterion, optimizer=optimizer, best_model_path=best_model_path, use_wandb=use_wandb)
        model.load_state_dict(torch.load(best_model_path))
        test_model(model, test_loader, criterion, use_wandb=use_wandb)
