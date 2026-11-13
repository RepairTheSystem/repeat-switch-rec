"""
Правила смешивания двух голов — общие для эвристик и нейросети, чтобы сравнение было честным.

Components:
  rep (U, N) — распределение repeat-головы по товарам истории (0 вне истории);
  exp (U, N) — распределение explore-головы по товарам НЕ из истории (0 на истории);
  p   (U,)   — вероятность режима повторов на этот запрос.

Правила:
  switch(p)  — мягкая смесь p·rep + (1−p)·exp (выученный переключатель);
  quota(q)   — q лучших из repeat-головы + остальное из explore (q общий или свой на пользователя);
  route(thr) — жёсткий роутинг: p > thr → вся выдача из repeat-головы, иначе — вся из explore
               (исходная идея «классификатор запроса → отдельная модель»).
"""
from dataclasses import dataclass

import numpy as np

from metrics import per_user, topk


@dataclass
class Components:
    rep: np.ndarray
    exp: np.ndarray
    p: np.ndarray


def switch(c, k, p=None):
    p = (c.p if p is None else np.asarray(p))[:, None]
    return topk(p * c.rep + (1 - p) * c.exp, k)


def quota(c, k, q):
    q = np.broadcast_to(np.asarray(q, dtype=float), (len(c.p),))
    rep_top, exp_top = topk(c.rep, k), topk(c.exp, k)
    n_pos = (c.rep > 0).sum(1)                 # меньше k повторов → добиваем explore
    out = np.empty((len(c.p), k), dtype=np.int64)
    for u in range(len(c.p)):
        m = int(min(q[u], n_pos[u], k))
        out[u, :m] = rep_top[u, :m]
        out[u, m:] = exp_top[u, :k - m]
    return out


def route(c, k, thr):
    return quota(c, k, np.where(c.p > thr, k, 0))


def tune(make_recs, grid, split, users, items, k):
    """Лучшее значение параметра по Recall@k на val. Возвращает (значение, recall)."""
    scores = [(g, per_user(make_recs(g), users, split, items, k).recall.mean()) for g in grid]
    return max(scores, key=lambda s: s[1])


def tune_quota(c, split, users, items, k):
    return tune(lambda q: quota(c, k, q), range(k + 1), split, users, items, k)[0]


def tune_route(c, split, users, items, k):
    grid = [-1.0, 2.0] + list(np.unique(np.quantile(c.p, np.linspace(0.05, 0.95, 19))))
    return tune(lambda t: route(c, k, t), grid, split, users, items, k)[0]
