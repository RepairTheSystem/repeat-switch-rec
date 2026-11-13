"""
Подбор learning rate на val (тест не используется): для каждой модели и lr — лучший val Recall@10 и эпоха.
  python -u lr_probe.py --regime high --seed 0 --lrs 1e-3 3e-3
"""
import argparse
import random

import numpy as np
import torch

from data import make_protocol, make_synthetic
from train import REGIMES, build_train_sequences, train_model

ap = argparse.ArgumentParser()
ap.add_argument("--regime", default="high", choices=list(REGIMES))
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-3])
ap.add_argument("--modes", nargs="+", default=["base", "switch"])
ap.add_argument("--max-epochs", type=int, default=40)
cli = ap.parse_args()

args = argparse.Namespace(k=10, max_len=50, dim=64, layers=2, patience=3, batch=128,
                          max_epochs=cli.max_epochs, device="cpu", lr=None)
cons, dur = REGIMES[cli.regime]
events, items = make_synthetic(consumable_share=cons, durable_rate=dur, seed=cli.seed)
train, val, _ = make_protocol(events)
seqs = build_train_sequences(train, args.max_len)
res = []
for mode in cli.modes:
    for lr in cli.lrs:
        random.seed(cli.seed); np.random.seed(cli.seed); torch.manual_seed(cli.seed)
        args.lr = lr
        _, ep, best = train_model(list(seqs), len(items), mode, args, val, val.users, items)
        res.append((mode, lr, ep, best))
        print(f"### {mode} lr={lr}: best val Recall@10 {best:.4f} at epoch {ep}")
print("\nИтог:")
for mode, lr, ep, best in res:
    print(f"  {mode:7s} lr={lr:g}: val Recall@10 {best:.4f}, эпоха {ep}")
