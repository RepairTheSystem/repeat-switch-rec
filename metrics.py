"""
Метрики. Главное — считать отдельно по двум типам целей:
  repeat  — пользователь купил то, что уже покупал;
  explore — купил новое.
Общая метрика прячет провал на одной из частей.

У пользователя может быть несколько целей (все покупки в окне), поэтому:
  Recall@k        — среднее по пользователям |hits| / |targets|;
  NDCG@k          — с несколькими релевантными, IDCG по min(|targets|, k);
  HR@k            — доля пользователей, у которых угадана хоть одна цель;
  Recall repeat / explore — микро-усреднение по целям своего типа: Σ hits / Σ targets.
"""
import numpy as np
import pandas as pd


def topk(scores, k):
    """Индексы k лучших по строкам, по убыванию."""
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, idx, 1), axis=1, kind="stable")
    return np.take_along_axis(idx, order, 1)


def true_repeat_frac(split, users):
    """Доля повторов среди целей каждого пользователя (нужна для оракула)."""
    return np.array([np.isin(split.targets[u], split.hist[u][0]).mean() for u in users])


def _membership(lists, n_items):
    """Ragged-списки товаров → плотная булева матрица (U, N)."""
    M = np.zeros((len(lists), n_items), dtype=bool)
    rows = np.repeat(np.arange(len(lists)), [len(x) for x in lists])
    M[rows, np.concatenate(lists).astype(np.int64)] = True
    return M


def per_user(recs, users, split, items, k=10):
    """recs: (U, >=k) — рекомендации для users (в том же порядке). Возвращает метрики по пользователям."""
    n_items = len(items)
    is_cons = items["is_consumable"].to_numpy().astype(float)
    has_truth = not np.isnan(is_cons).any()
    disc = 1.0 / np.log2(np.arange(2, k + 2))

    T = _membership([split.targets[u] for u in users], n_items)   # цели
    H = _membership([split.hist[u][0] for u in users], n_items)   # история
    top = np.asarray(recs)[:, :k]
    r = np.arange(len(users))[:, None]
    hit = T[r, top]                                  # (U, k): позиция выдачи попала в цель
    in_hist = H[r, top]
    n_t = T.sum(1)
    df = pd.DataFrame({
        "user": users, "n_t": n_t, "n_rep_t": (T & H).sum(1),
        "hits": hit.sum(1), "hits_rep": (hit & in_hist).sum(1), "hits_exp": (hit & ~in_hist).sum(1),
        "ndcg": (hit * disc).sum(1) / np.cumsum(disc)[np.minimum(n_t, k) - 1],
        "n_rep_in_top": in_hist.sum(1),
        "wasted": (in_hist & (is_cons[top] == 0)).sum(1) if has_truth else np.nan,
    })
    df["recall"] = df.hits / df.n_t
    df["hit"] = (df.hits > 0).astype(float)
    df["repeat_frac"] = df.n_rep_t / df.n_t
    return df


def auc(score, label):
    """ROC AUC через ранги (Манн–Уитни)."""
    label = np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    r = pd.Series(score).rank().to_numpy()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def summarize(df, p_switch=None, k=10):
    n_exp_t = (df.n_t - df.n_rep_t).sum()
    out = {
        f"Recall@{k}": df.recall.mean(),
        f"NDCG@{k}": df.ndcg.mean(),
        f"HR@{k}": df.hit.mean(),
        "Recall repeat": df.hits_rep.sum() / max(df.n_rep_t.sum(), 1),
        "Recall explore": df.hits_exp.sum() / max(n_exp_t, 1),
        "repeat share (true)": df.n_rep_t.sum() / df.n_t.sum(),
        "repeat slots": df.n_rep_in_top.mean() / k,
        "wasted slots": df.wasted.mean(),
    }
    if p_switch is not None:
        # p_switch — прогноз «следующая покупка будет повтором»; правда — доля повторов в целях
        p = np.asarray(p_switch)
        y = (df.repeat_frac >= 0.5).to_numpy()
        out["switch AUC"] = auc(p, y)
        out["switch Brier"] = float(np.mean((p - df.repeat_frac.to_numpy()) ** 2))
        out["p | repeat"] = p[y].mean() if y.any() else np.nan
        out["p | explore"] = p[~y].mean() if (~y).any() else np.nan
    return out


def bootstrap_diff(a, b, col="recall", n_boot=2000, seed=0):
    """Парный бутстрап по пользователям: среднее(a − b) и 95% доверительный интервал."""
    d = a[col].to_numpy() - b[col].to_numpy()
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
