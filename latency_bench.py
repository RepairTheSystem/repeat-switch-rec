"""
Бенчмарк латентности: не зависит от весов, поэтому модели не обучаются — меряется только архитектура
и полный путь выдачи (паддинг → прямой проход → смешивание голов → top-k) на реальных историях.

  python latency_bench.py --device cpu
  python latency_bench.py --device cuda --out results/latency_cuda.csv

Сценарии:
  online — batch=1 (один запрос = один пользователь), перцентили p50/p95/p99 по --n запросам;
  batch  — все тестовые пользователи батчами по --batch, мс на пользователя.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

import baselines as bl
from data import make_protocol, make_synthetic
from heuristics import heur_components
from mixing import switch
from model import OneStageRec
from train import REGIMES, SWITCH_VARIANTS, model_kwargs, nn_outputs, online_latency, repeat_prior, timed, variant_prefix
from metrics import topk

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--regime", default="high", choices=list(REGIMES))
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads для CPU")
ap.add_argument("--variants", nargs="+", default=list(SWITCH_VARIANTS), choices=list(SWITCH_VARIANTS))
ap.add_argument("--out", default=None)
cli = ap.parse_args()
if cli.threads:
    torch.set_num_threads(cli.threads)

args = argparse.Namespace(k=10, max_len=50, dim=64, layers=2, batch=cli.batch, device=cli.device)
cons, dur = REGIMES[cli.regime]
events, items = make_synthetic(consumable_share=cons, durable_rate=dur, seed=0)
train, val, test = make_protocol(events)
tu, n_items, k = test.users, len(items), 10
prior = repeat_prior(train, n_items)
print(f"device={cli.device} threads={torch.get_num_threads()} test users={len(tu)}")

rows = []
for var, fn in (("freq+pop", lambda: bl.freq_pop(test, tu, n_items, k)),
                ("tifu-knn", lambda: bl.tifu_knn(test, tu, n_items, k)),
                ("heur/switch", lambda: switch(heur_components(test, tu, items)[1], k))):
    _, sec = timed(fn, args)
    rows.append({"variant": var, "batch ms/user": 1000 * sec / len(tu)})

models = {"nn/base": OneStageRec(n_items, mode="base").to(cli.device)}
for sv in cli.variants:
    kw = model_kwargs(sv, prior)
    full = kw.pop("full_mask", False)
    m = OneStageRec(n_items, mode="switch", **kw).to(cli.device)
    m.full_mask = full
    models[f"{variant_prefix(sv)}/switch"] = m

for name, m in models.items():
    m.eval()
    nn_outputs(m, test, tu[:args.batch], args)                       # прогрев
    out, sec = timed(lambda: nn_outputs(m, test, tu, args), args)
    sec += timed(lambda: topk(out, k) if m.mode == "base" else switch(out, k), args)[1]
    rows.append({"variant": name, "params": sum(p.numel() for p in m.parameters()),
                 "batch ms/user": 1000 * sec / len(tu), **online_latency(m, test, tu, args, n=cli.n)})
    print(rows[-1])

df = pd.DataFrame(rows).set_index("variant")
df.insert(0, "device", cli.device)
with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
    print(df)
if cli.out:
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    df.to_csv(cli.out)
