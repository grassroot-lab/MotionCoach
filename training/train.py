import argparse
import math
import os
import random
from collections import Counter
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from dataset_kp import KeypointWindowDataset
from model_tcn import TCNMultiTask


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def confusion_and_metrics(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, float, float]:
    """
    Returns:
      confusion matrix [C,C],
      accuracy,
      macro_f1
    """
    y_pred = logits.argmax(dim=1)
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=logits.device)
    for t, p in zip(y_true.view(-1), y_pred.view(-1)):
        conf[t.long(), p.long()] += 1

    correct = (y_pred == y_true).sum().item()
    total = y_true.numel()
    acc = correct / max(1, total)

    # Macro F1
    eps = 1e-9
    tp = torch.diag(conf).float()
    fp = conf.sum(dim=0).float() - tp
    fn = conf.sum(dim=1).float() - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    macro_f1 = f1.mean().item()
    return conf, acc, macro_f1


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    cnt = Counter(labels)
    weights = []
    total = sum(cnt.values())
    for c in range(num_classes):
        # inverse frequency with normalization
        w = total / (cnt.get(c, 0) + 1e-6)
        weights.append(w)
    w = torch.tensor(weights, dtype=torch.float32)
    w = w / w.mean()
    return w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="")

    parser.add_argument("--num_classes", type=int, default=3)
    parser.add_argument("--num_joints", type=int, default=17)
    parser.add_argument("--num_channels", type=int, default=3)  # (x,y,conf)
    parser.add_argument("--T", type=int, default=48)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--num_levels", type=int, default=4)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--use_regression", action="store_true")
    parser.add_argument("--lambda_reg", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    train_ds = KeypointWindowDataset(
        manifest_path=args.train_manifest,
        data_root=args.data_root,
        T=args.T,
        num_joints=args.num_joints,
        num_channels=args.num_channels,
        normalize=True,
        use_conf=True,
    )
    val_ds = KeypointWindowDataset(
        manifest_path=args.val_manifest,
        data_root=args.data_root,
        T=args.T,
        num_joints=args.num_joints,
        num_channels=args.num_channels,
        normalize=True,
        use_conf=True,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Class weights for imbalance
    all_train_labels = [r.label for r in train_ds.rows]
    class_weights = compute_class_weights(all_train_labels, args.num_classes).to(device)

    model = TCNMultiTask(
        num_classes=args.num_classes,
        num_frames=args.T,
        num_joints=args.num_joints,
        num_channels=args.num_channels,
        hidden_channels=args.hidden_channels,
        num_levels=args.num_levels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        enable_regression=args.use_regression,
    ).to(device)

    # Losses
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
    reg_criterion = nn.SmoothL1Loss(beta=1.0)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_macro_f1 = -1.0
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0

        for x, y, s in train_loader:
            x = x.to(device)  # [B,T,J,C]
            y = y.to(device=device, dtype=torch.long)
            s = None if s is None else s.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, score_pred = model(x)

            loss_cls = cls_criterion(logits, y)
            loss = loss_cls

            if args.use_regression and s is not None and score_pred is not None:
                loss_reg = reg_criterion(score_pred, s)
                loss = loss_cls + args.lambda_reg * loss_reg

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_seen += x.size(0)

        scheduler.step()

        avg_loss = total_loss / max(1, total_seen)

        # Validation
        model.eval()
        val_total_acc = 0.0
        val_total_f1 = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y, s in val_loader:
                x = x.to(device)
                y = y.to(device=device, dtype=torch.long)

                logits, score_pred = model(x)
                conf, acc, macro_f1 = confusion_and_metrics(logits, y, args.num_classes)
                val_total_acc += acc
                val_total_f1 += macro_f1
                val_batches += 1

        val_acc = val_total_acc / max(1, val_batches)
        val_macro_f1 = val_total_f1 / max(1, val_batches)

        print(
            f"[Epoch {epoch:03d}] train_loss={avg_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            ckpt_path = f"checkpoints/best_tcn_epoch{epoch}_f1{val_macro_f1:.3f}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": val_macro_f1,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    print(f"[DONE] best_val_macro_f1={best_val_macro_f1:.4f}")


if __name__ == "__main__":
    main()

