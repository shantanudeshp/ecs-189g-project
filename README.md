# ECS189G Project: AWS End-to-End Commands

This README collects the exact commands used to run the full pipeline on an AWS EC2 GPU instance.

## 0) Upload repo to EC2 (exclude venv/cache/.git)
Run from your **local machine**:
```bash
rsync -avz \
  --exclude ".venv/" \
  --exclude ".git/" \
  --exclude ".ipynb_checkpoints/" \
  --exclude ".git/lfs/cache/" \
  --exclude ".DS_Store" \
  -e "ssh -i ~/gege.pem" \
  /Users/gegekang/Desktop/ecs-189g-project/ \
  gege@18.188.99.104:~/ecs-189g-project/
```

## 1) SSH in + base setup
```bash
ssh -i ~/gege.pem gege@18.188.99.104
cd ~/ecs-189g-project

sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip git

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2) Install Python deps
```bash
pip install -r requirements.txt
```

Install PyTorch with CUDA (adjust CUDA wheel if needed):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you see a `device_map` error, install accelerate:
```bash
pip install accelerate
```

Optional (for faster HF downloads):
```bash
export HF_TOKEN=YOUR_TOKEN
```

## 3) Collect hidden states + labels
```bash
python load_classifier.py \
  --model distilgpt2 \
  --dataset nq_open \
  --balanced \
  --n-train 500 --n-val 100 --n-test 100 \
  --layers 0,last \
  --device cuda
```

Outputs:
```
datasets/nq_open/distilgpt2/hs/{train,val,test}/layer_*.pth
```

## 4) Train classifier (layer 6)
```bash
python cls.py distilgpt2 nq_open 0 \
  --layers 6 \
  --label-key labels_strict \
  --device cuda
```

Outputs:
```
clss/nq_open/distilgpt2/label_labels_strict__m_0_l_6_b_128.pth
```

## 5) Build intervention dataset
```bash
python build_intervention_dataset.py \
  --model distilgpt2 \
  --dataset nq_open \
  --n-train 500 --n-val 100 --n-test 100 \
  --layers last \
  --device cuda
```

Outputs:
```
datasets/nq_open/distilgpt2/hs/{train_fct,val_fct,test_fct}/layer_6.pth
```

## 6) Train intervention model (g_phi)
```bash
python train_gphi_mse.py \
  --model distilgpt2 \
  --dataset nq_open \
  --layer 6 \
  --device cuda
```

Outputs:
```
gphi_ckpts/nq_open/distilgpt2/layer_6/gphi_mse.pth
```

## 7) Run intervention decoding
```bash
python intervene_decode.py \
  --model distilgpt2 \
  --prompt "Question: Who wrote The Hobbit?\nAnswer:" \
  --cls-ckpt clss/nq_open/distilgpt2/label_labels_strict__m_0_l_6_b_128.pth \
  --gphi-ckpt gphi_ckpts/nq_open/distilgpt2/layer_6/gphi_mse.pth \
  --layer 6 \
  --mode 0 \
  --alpha 0.3 \
  --device cuda
```

## 8) Download results back to local (optional)
```bash
rsync -avz -e "ssh -i ~/gege.pem" \
  gege@18.188.99.104:~/ecs-189g-project/clss/nq_open/distilgpt2/ \
  /Users/gegekang/Desktop/ecs-189g-project/clss/nq_open/distilgpt2/
```
