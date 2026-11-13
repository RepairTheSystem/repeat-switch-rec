"""
Эксперимент: одностадийная модель без режимов vs квота повторов vs выученный переключатель
vs жёсткий роутинг, против сильных простых бейзлайнов.

Протокол (см. data.make_protocol):
  * модель учится на событиях < t_val, ранняя остановка и подбор квоты/порога/гиперпараметров
    бейзлайнов — на val-окне; на тесте ничего не подбирается;
  * запрос — в начале окна, цели — все покупки в окне (время покупки модели неизвестно);
  * несколько сидов (сид меняет и данные, и инициализацию), парный бутстрап по пользователям.

Запуск:
  python train.py                                         # 3 режима доли повторов × 3 сида
  python train.py --regimes high --seeds 0                # один режим, быстро
  python train.py --protocol next --seeds 0 --out results/leak_next   # старый протокол с утечкой
  python train.py --skip-nn                               # быстро, только эвристики и бейзлайны
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch

import baselines as bl
from data import make_protocol, make_synthetic
from heuristics import heur_components
from metrics import bootstrap_diff, per_user, summarize, topk, true_repeat_frac
from mixing import Components, quota, route, switch, tune, tune_quota, tune_route
from model import OneStageRec

# режимы синтетики: (consumable_share, durable_rate) → доля повторов среди целей теста
REGIMES = {"low": (0.10, 15.0),     # 0.13–0.37 (по сидам 0–2)
           "mid": (0.15, 15.0),     # 0.23–0.54
           "high": (0.40, 0.5)}     # 0.82–0.88 (исходная синтетика)

# --------------------------------------------------------------------------- нейросеть


def build_train_sequences(train, max_len):
    """
    Окна по max_len+1 покупок с конца истории; каждый переход попадает в обучение ровно один раз.
    Третий элемент — товары, купленные до окна (для маски explore-головы, см. model.forward).
    """
    seqs = []
    for _, g in train.groupby("user"):
        it = g["item"].to_numpy() + 1             # +1: 0 зарезервирован под паддинг
        ts = g["ts"].to_numpy()
        end = len(it)
        while end >= 2:
            start = max(0, end - (max_len + 1))
            seqs.append((it[start:end], ts[start:end], np.unique(it[:start])))
            if start == 0:
                break
            end = start + 1
    return seqs


def pad_batch(batch, length, n_items=None):
    """
    Паддинг справа. Время в паддинге = последнее реальное время (чтобы не было мусорных интервалов).
    Если задан n_items — ещё и маска (B, n_items + 1) товаров, купленных до окна.
    """
    items = np.zeros((len(batch), length), dtype=np.int64)
    times = np.zeros((len(batch), length), dtype=np.float32)
    lens = np.zeros(len(batch), dtype=np.int64)
    pre = np.zeros((len(batch), n_items + 1), dtype=bool) if n_items is not None else None
    for b, (it, ts, before) in enumerate(batch):
        n = len(it)
        items[b, :n] = it
        times[b, :n] = ts
        times[b, n:] = ts[-1]
        lens[b] = n
        if pre is not None:
            pre[b, before] = True
    pre = torch.from_numpy(pre) if pre is not None else None
    return torch.from_numpy(items), torch.from_numpy(times), torch.from_numpy(lens), pre


@torch.no_grad()
def nn_outputs(model, split, users, args):
    """Прогноз на момент запроса. base → scores (U, N); switch → Components."""
    model.eval()
    outs = []
    for s in range(0, len(users), args.batch):
        bu = users[s:s + args.batch]
        batch = [(split.hist[u][0][-args.max_len:] + 1, split.hist[u][1][-args.max_len:],
                  np.unique(split.hist[u][0][:-args.max_len] + 1)) for u in bu]
        n_mask = model.n_items if getattr(model, "full_mask", False) else None
        items, times, lens, pre = pad_batch(batch, args.max_len, n_mask)
        t_now = torch.tensor([split.t_query[u] for u in bu], dtype=torch.float32)
        t_now = t_now.unsqueeze(1).expand_as(times).contiguous()
        items, times, t_now = items.to(args.device), times.to(args.device), t_now.to(args.device)
        pre = pre.to(args.device) if pre is not None else None
        logp, p_rep, p_exp, p_sw = model(items, times, t_now, pre)
        rows, last = torch.arange(len(bu), device=args.device), (lens - 1).to(args.device)
        if model.mode == "base":
            outs.append(logp[rows, last, 1:].cpu().numpy())
        else:
            outs.append((p_rep[rows, last, 1:].cpu().numpy(), p_exp[rows, last, 1:].cpu().numpy(),
                         p_sw[rows, last, 0].cpu().numpy()))
    if model.mode == "base":
        return np.concatenate(outs)
    return Components(*(np.concatenate([o[i] for o in outs]) for i in range(3)))


def nn_recs(model, split, users, args):
    out = nn_outputs(model, split, users, args)
    return topk(out, args.k) if model.mode == "base" else switch(out, args.k)


def repeat_prior(train, n_items, alpha=5.0):
    """
    Логит сглаженной доли повторных покупок товара по train: (повторы + α·g) / (покупки + α),
    повторы = покупки − число различных покупателей, g — общая доля. Индекс 0 — паддинг.
    """
    n = np.bincount(train["item"], minlength=n_items).astype(float)
    buyers = train.groupby("item")["user"].nunique().reindex(range(n_items), fill_value=0).to_numpy()
    rep = n - buyers
    g = rep.sum() / max(n.sum(), 1.0)
    r = np.clip((rep + alpha * g) / (n + alpha), 1e-4, 1 - 1e-4)
    return np.concatenate([[0.0], np.log(r / (1 - r))])


# варианты switch-модели для абляции: имя → опции (см. model.py)
#   mask  — explore-голова исключает ВСЮ историю, а не только окно последних max_len покупок
#           (главная причина wasted slots: 98% из них — товары, выпавшие из окна);
#   bias / null / prior — опции repeat-головы.
_ALL = {"rep_bias": True, "rep_null": True, "rep_prior": True}
SWITCH_VARIANTS = {"switch": {}, "mask": {"full_mask": True},
                   "mask+bias": {"full_mask": True, "rep_bias": True},
                   "mask+null": {"full_mask": True, "rep_null": True},
                   "mask+prior": {"full_mask": True, "rep_prior": True},
                   "mask+all": {"full_mask": True, **_ALL}}


def variant_prefix(sv):
    return "nn" if sv == "switch" else f"nn+{sv}"


def model_kwargs(sv, prior):
    kw = dict(SWITCH_VARIANTS[sv])
    if kw.pop("rep_prior", False):
        kw["rep_prior"] = prior
    return kw


def train_model(seqs, n_items, mode, args, val, val_users, items, full_mask=False, **model_kw):
    """Ранняя остановка по Recall@k на val-окне; возвращается лучшая эпоха."""
    model = OneStageRec(n_items, mode=mode, d=args.dim, n_layers=args.layers,
                        max_len=args.max_len, **model_kw).to(args.device)
    model.full_mask = full_mask
    lr = args.lr if args.lr is not None else getattr(args, f"lr_{mode}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, best_epoch, bad = -1.0, None, 0, 0
    for epoch in range(args.max_epochs):
        model.train()
        random.shuffle(seqs)
        total, count = 0.0, 0
        for s in range(0, len(seqs), args.batch):
            items_b, times, _, pre = pad_batch(seqs[s:s + args.batch], args.max_len + 1,
                                               n_items if full_mask else None)
            items_b, times = items_b.to(args.device), times.to(args.device)
            pre = pre.to(args.device) if pre is not None else None
            inp, tgt = items_b[:, :-1], items_b[:, 1:]
            logp, _, _, _ = model(inp, times[:, :-1], times[:, 1:], pre)
            lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            mask = (tgt != 0).float()
            loss = -(lp * mask).sum() / mask.sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * mask.sum().item()
            count += mask.sum().item()
        rec = per_user(nn_recs(model, val, val_users, args), val_users, val, items, args.k).recall.mean()
        print(f"    [{mode}] epoch {epoch + 1}  train NLL {total / count:.4f}  val Recall@{args.k} {rec:.4f}")
        if rec > best + 1e-4:
            best, best_state, best_epoch, bad = rec, copy.deepcopy(model.state_dict()), epoch + 1, 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_epoch == args.max_epochs:
        print(f"    [{mode}] ВНИМАНИЕ: лучшая эпоха = лимит {args.max_epochs}, модель могла не сойтись")
    model.load_state_dict(best_state)
    return model, best_epoch, best


# --------------------------------------------------------------------------- варианты


def two_head_variants(prefix, base_scores, comps, split, users, k, q, thr, full=True):
    """Все правила смешивания поверх одних и тех же голов (full=False — только главные, для абляции)."""
    frac = true_repeat_frac(split, users)
    v = {}
    if base_scores is not None:
        v[f"{prefix}/base"] = (topk(base_scores, k), None)
    if full:
        v[f"{prefix}/fixed30"] = (quota(comps, k, round(0.3 * k)), None)
    v[f"{prefix}/fixed-tuned"] = (quota(comps, k, q), None)
    v[f"{prefix}/switch"] = (switch(comps, k), comps.p)
    if full:
        v[f"{prefix}/route"] = (route(comps, k, thr), comps.p)
    v[f"{prefix}/oracle-switch"] = (switch(comps, k, p=frac), None)    # знает долю повторов в целях
    if full:
        v[f"{prefix}/oracle-quota"] = (quota(comps, k, np.round(k * frac)), None)
    return v


# --------------------------------------------------------------------------- латентность


def _sync(args):
    if str(args.device).startswith("cuda"):
        torch.cuda.synchronize()


def timed(fn, args):
    """(результат, секунды) — с синхронизацией GPU."""
    _sync(args)
    t = time.perf_counter()
    out = fn()
    _sync(args)
    return out, time.perf_counter() - t


def online_latency(model, split, users, args, n=200, warmup=20, seed=0):
    """
    Онлайн-сценарий: один запрос = один пользователь (batch=1), полный путь от истории
    до топ-k: паддинг → прямой проход → смешивание голов → top-k. Возвращает p50/p95/p99, мс.
    """
    one = argparse.Namespace(**{**vars(args), "batch": 1})
    sample = np.random.default_rng(seed).choice(users, n + warmup)
    ms = []
    for i, u in enumerate(sample):
        _, dt = timed(lambda: nn_recs(model, split, np.array([u]), one), args)
        if i >= warmup:
            ms.append(dt * 1000)
    return {f"online p{q} ms": float(np.percentile(ms, q)) for q in (50, 95, 99)}


def run_one(regime, seed, args):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    k = args.k
    cons, dur = REGIMES[regime]
    events, items = make_synthetic(n_users=args.n_users, consumable_share=cons, durable_rate=dur,
                                   seed=seed)
    n_items = len(items)
    train, val, test = make_protocol(events, args.protocol, args.t_val, args.t_test, args.window, args.gap)
    vu, tu = val.users, test.users
    print(f"\n=== regime={regime} seed={seed}: events {len(events)}, train {len(train)}, "
          f"val users {len(vu)}, test users {len(tu)}")

    chosen = {}
    V = {"val": {}, "test": {}}      # split -> name -> (recs, p_switch)
    lat = []                          # латентность и стоимость
    base_row = {"regime": regime, "seed": seed, "device": str(args.device), "users": len(tu)}

    def lat_batch(name, sec, **extra):
        lat.append({**base_row, "variant": name, "batch ms/user": 1000 * sec / len(tu), **extra})

    # --- простые бейзлайны (время — полный расчёт выдачи для всех тестовых пользователей)
    g, _ = tune(lambda g: bl.tifu_knn(val, vu, n_items, k, *g), bl.TIFU_GRID, val, vu, items, k)
    chosen["tifu (tau, K, alpha)"] = g
    for name, sp, us in (("val", val, vu), ("test", test, tu)):
        for var, fn in (("pop", lambda: bl.pop(sp, us, n_items, k)),
                        ("freq+pop", lambda: bl.freq_pop(sp, us, n_items, k)),
                        ("tifu-knn", lambda: bl.tifu_knn(sp, us, n_items, k, *g))):
            recs, sec = timed(fn, args)
            V[name][var] = (recs, None)
            if name == "test":
                lat_batch(var, sec)

    # --- эвристические головы
    hb_v, hc_v = heur_components(val, vu, items)
    (hb_t, hc_t), sec = timed(lambda: heur_components(test, tu, items), args)
    lat_batch("heur/switch", sec + timed(lambda: switch(hc_t, k), args)[1])
    q, thr = tune_quota(hc_v, val, vu, items, k), tune_route(hc_v, val, vu, items, k)
    chosen["heur q"], chosen["heur thr"] = q, float(thr)
    V["val"].update(two_head_variants("heur", hb_v, hc_v, val, vu, k, q, thr))
    V["test"].update(two_head_variants("heur", hb_t, hc_t, test, tu, k, q, thr))

    # --- нейросеть: base + варианты switch-модели (абляция repeat-головы)
    if not args.skip_nn:
        seqs = build_train_sequences(train, args.max_len)
        prior = repeat_prior(train, n_items)
        models = {}
        t0 = time.time()
        models["base"], chosen["nn/base epoch"], _ = train_model(list(seqs), n_items, "base", args, val, vu, items)
        for sv in args.switch_variants:
            models[sv], chosen[f"{variant_prefix(sv)} epoch"], _ = train_model(
                list(seqs), n_items, "switch", args, val, vu, items, **model_kwargs(sv, prior))
        chosen["nn train sec"] = round(time.time() - t0)

        nb_v = nn_outputs(models["base"], val, vu, args)
        nb_t, sec = timed(lambda: nn_outputs(models["base"], test, tu, args), args)
        lat_batch("nn/base", sec + timed(lambda: topk(nb_t, k), args)[1],
                  params=sum(p.numel() for p in models["base"].parameters()),
                  **online_latency(models["base"], test, tu, args, seed=seed))
        for sv in args.switch_variants:
            m, pre = models[sv], variant_prefix(sv)
            nc_v = nn_outputs(m, val, vu, args)
            nc_t, sec = timed(lambda: nn_outputs(m, test, tu, args), args)
            lat_batch(f"{pre}/switch", sec + timed(lambda: switch(nc_t, k), args)[1],
                      params=sum(p.numel() for p in m.parameters()),
                      **online_latency(m, test, tu, args, seed=seed))
            q, thr = tune_quota(nc_v, val, vu, items, k), tune_route(nc_v, val, vu, items, k)
            chosen[f"{pre} q"], chosen[f"{pre} thr"] = q, float(thr)
            full = sv == "switch"
            V["val"].update(two_head_variants(pre, nb_v if full else None, nc_v, val, vu, k, q, thr, full))
            V["test"].update(two_head_variants(pre, nb_t if full else None, nc_t, test, tu, k, q, thr, full))

    # --- метрики
    rows, per = [], {"val": {}, "test": {}}
    for name, sp, us in (("val", val, vu), ("test", test, tu)):
        for var, (recs, p) in V[name].items():
            df = per_user(recs, us, sp, items, k)
            per[name][var] = df
            rows.append({"regime": regime, "seed": seed, "split": name, "variant": var,
                         **summarize(df, p, k)})

    # --- значимость: эталон против конкурентов (лучший не-нейросетевой выбран по val, не по test)
    ref = "nn/switch" if not args.skip_nn else "heur/switch"
    non_nn = [v for v in per["val"] if not v.startswith("nn") and "oracle" not in v]
    best_simple = max(non_nn, key=lambda v: per["val"][v].recall.mean())
    rivals = [best_simple] + [v for v in ("nn/base", "nn/fixed-tuned", "nn/route", "heur/switch")
                              if v in per["test"] and v != ref and v != best_simple]
    pairs = [(ref, rv, col) for rv in rivals for col in ("recall", "ndcg")]
    # абляция: каждый вариант repeat-головы против исходного switch, в т.ч. по потраченным слотам
    pairs += [(f"{variant_prefix(sv)}/switch", ref, col) for sv in getattr(args, "switch_variants", [])
              if sv != "switch" and not args.skip_nn for col in ("recall", "ndcg", "wasted")]
    boot = []
    for a, b, col in pairs:
        d, lo, hi = bootstrap_diff(per["test"][a], per["test"][b], col, seed=seed)
        boot.append({"regime": regime, "seed": seed, "ref": a, "rival": b,
                     "rival is best simple (val)": b == best_simple,
                     "metric": col, "diff": d, "ci_lo": lo, "ci_hi": hi})
    print("    подобрано на val:", chosen)
    return rows, boot, {"regime": regime, "seed": seed, **{k_: str(v) for k_, v in chosen.items()}}, lat


# --------------------------------------------------------------------------- отчёт

SHOW = ["Recall@10", "NDCG@10", "Recall repeat", "Recall explore", "repeat slots", "wasted slots",
        "switch AUC", "switch Brier"]


def to_md(df, **kw):
    """Markdown-таблица; без пакета tabulate — обычный текст."""
    try:
        return df.to_markdown(**kw)
    except ImportError:
        return "```\n" + df.to_string(index=kw.get("index", True)) + "\n```"


def fmt_table(df, cols):
    g = df.groupby(["regime", "variant"], sort=False)
    mean, std = g[cols].mean(), g[cols].std().fillna(0)
    out = mean.copy().astype(object)
    for c in cols:
        out[c] = [("—" if np.isnan(m) else f"{m:.3f} ± {s:.3f}") for m, s in zip(mean[c], std[c])]
    return out


def report(rows, boot, chosen, lat, args):
    os.makedirs(args.out, exist_ok=True)
    res, bt, ch, lt = pd.DataFrame(rows), pd.DataFrame(boot), pd.DataFrame(chosen), pd.DataFrame(lat)
    res.to_csv(os.path.join(args.out, "metrics.csv"), index=False)
    bt.to_csv(os.path.join(args.out, "bootstrap.csv"), index=False)
    ch.to_csv(os.path.join(args.out, "chosen.csv"), index=False)
    lt.to_csv(os.path.join(args.out, "latency.csv"), index=False)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    test = res[res.split == "test"]
    k = args.k
    cols = [c.replace("@10", f"@{k}") for c in SHOW if c.replace("@10", f"@{k}") in test.columns]
    lines = [f"# Результаты: protocol={args.protocol}, window={args.window}, seeds={args.seeds}\n",
             "Тест; среднее ± std по сидам. Квота/порог/параметры TIFU подобраны на val.\n"]
    rs = test.groupby("regime", sort=False)["repeat share (true)"].mean()
    lines.append("Доля повторов среди целей: " + ", ".join(f"{s}: {v:.2f}" for s, v in rs.items()) + "\n")
    lines.append(to_md(fmt_table(test, cols)))
    if len(bt):
        lines.append("\n\n## Парный бутстрап на тесте: ref минус rival\n")
        lines.append("CI — средние по сидам границы 95% интервала; `CI>0` / `CI<0` — в скольких сидах интервал "
                     "целиком выше / ниже нуля. Для `wasted` лучше отрицательная разница.\n")
        for ref, part in bt.groupby("ref", sort=False):
            b = part.groupby(["regime", "rival", "metric"], sort=False).agg(
                diff=("diff", "mean"), ci_lo=("ci_lo", "mean"), ci_hi=("ci_hi", "mean"),
                **{"CI>0": ("ci_lo", lambda x: f"{(x > 0).sum()}/{len(x)}"),
                   "CI<0": ("ci_hi", lambda x: f"{(x < 0).sum()}/{len(x)}")})
            lines.append(f"\n### {ref}\n")
            lines.append(to_md(b, floatfmt=".4f"))
    if len(lt):
        lc = [c for c in ("params", "batch ms/user", "online p50 ms", "online p95 ms", "online p99 ms") if c in lt]
        L = lt.groupby("variant", sort=False)[lc].mean()
        lines.append(f"\n\n## Латентность и стоимость (устройство: {args.device}; среднее по прогонам)\n")
        lines.append("`batch ms/user` — полный расчёт выдачи для всех тестовых пользователей, делённый на их число "
                     f"(батч {args.batch} для нейросетей). `online` — один запрос = один пользователь, "
                     "от истории до топ-k, перцентили по 200 запросам.\n")
        lines.append(to_md(L, floatfmt=".3f"))
    lines.append("\n\n## Подобранные на val параметры\n")
    lines.append(to_md(ch, index=False))
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=["high", "mid", "low"], choices=list(REGIMES),
                    help="режимы синтетики по доле повторов (см. REGIMES)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-users", type=int, default=4000)
    ap.add_argument("--protocol", default="window", choices=["window", "next", "next-start"])
    ap.add_argument("--t-val", type=float, default=351.0)
    ap.add_argument("--t-test", type=float, default=358.0)
    ap.add_argument("--window", type=float, default=7.0)
    ap.add_argument("--gap", type=float, default=0.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=None, help="общий lr; иначе --lr-base / --lr-switch")
    ap.add_argument("--lr-base", type=float, default=1e-3)
    ap.add_argument("--lr-switch", type=float, default=1e-3)
    ap.add_argument("--switch-variants", nargs="+", default=list(SWITCH_VARIANTS), choices=list(SWITCH_VARIANTS),
                    help="какие варианты repeat-головы обучать (абляция); 'switch' — исходный, обязателен")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if "switch" not in args.switch_variants:
        args.switch_variants = ["switch"] + args.switch_variants

    rows, boot, chosen, lat = [], [], [], []
    for regime in args.regimes:
        for seed in args.seeds:
            r, b, c, t = run_one(regime, seed, args)
            rows += r
            boot += b
            chosen.append(c)
            lat += t
    report(rows, boot, chosen, lat, args)


if __name__ == "__main__":
    main()
