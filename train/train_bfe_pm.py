"""Train bfe-pm (DualStreamBERT, wide_struct=True) — reproduces paper results.

Usage:
    python train/train_bfe_pm.py \\
        --data data/processed.parquet \\
        --seed 42 \\
        --out_dir checkpoints/

    # Run all 3 seeds
    for seed in 0 1 42; do
        python train/train_bfe_pm.py --data data/processed.parquet --seed $seed --out_dir checkpoints/
    done
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

sys.path.insert(0, str(Path(__file__).parent.parent))
from bfe_pm.model import DualStreamBERT
from train.dataset import load_splits

BERT_NAME = "trueto/medbert-kd-chinese"


def eval_loop(model, loader, scaler_y, device):
    import numpy as np
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for ids, mask, struct, labels in loader:
            p = model(ids.to(device), mask.to(device), struct.to(device))
            preds.append(p.cpu().numpy())
            targets.append(labels.numpy())
    p_arr = scaler_y.inverse_transform(__import__("numpy").concatenate(preds).reshape(-1, 1)).flatten()
    t_arr = scaler_y.inverse_transform(__import__("numpy").concatenate(targets).reshape(-1, 1)).flatten()
    mae = float(__import__("numpy").mean(__import__("numpy").abs(p_arr - t_arr)))
    return {"mae": mae}


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Seed: {args.seed}")

    train_ldr, val_ldr, test_ldr, scaler_y = load_splits(
        args.data, tokenizer_name=BERT_NAME, seed=args.seed,
        batch_size=args.batch_size, max_len=128,
    )

    model = DualStreamBERT(bert_name=BERT_NAME, freeze_layers=8, wide_struct=True).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")

    bert_params  = list(model.bert.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("bert.") and p.requires_grad]
    optimizer = AdamW([
        {"params": bert_params,  "lr": args.bert_lr},
        {"params": other_params, "lr": args.mlp_lr},
    ], weight_decay=0.01)
    scheduler = OneCycleLR(
        optimizer, max_lr=[args.bert_lr, args.mlp_lr],
        total_steps=args.epochs * len(train_ldr), pct_start=0.1,
    )
    loss_fn = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = out_dir / f"bfe_pm_seed{args.seed}_best.pt"
    best_val_mae = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        for ids, mask, struct, labels in train_ldr:
            optimizer.zero_grad()
            loss = loss_fn(model(ids.to(device), mask.to(device), struct.to(device)), labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        val_m = eval_loop(model, val_ldr, scaler_y, device)
        print(f"[E{epoch:02d}] val_mae={val_m['mae']:.2f} ({time.time()-t0:.1f}s)")

        if val_m["mae"] < best_val_mae:
            best_val_mae = val_m["mae"]
            torch.save({"model_state": model.state_dict(), "scaler_y": scaler_y}, best_ckpt)

    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    test_m = eval_loop(model, test_ldr, scaler_y, device)
    print(f"\nTest MAE: {test_m['mae']:.2f} min  |  Best ckpt: {best_ckpt}")
    with open(out_dir / f"bfe_pm_seed{args.seed}_results.json", "w") as f:
        json.dump({"seed": args.seed, "best_val_mae": best_val_mae, "test": test_m}, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data",       required=True)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out_dir",    default="checkpoints")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--bert_lr",    type=float, default=2e-5)
    p.add_argument("--mlp_lr",     type=float, default=1e-3)
    train(p.parse_args())
