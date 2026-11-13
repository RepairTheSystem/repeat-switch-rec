"""
Сильные простые бейзлайны. В задачах с повторными покупками они часто обгоняют нейросети
(см. Li et al., «A Next Basket Recommendation Reality Check»), поэтому без них вывод
«нейросеть лучше» ничего не стоит.

  pop       — популярность за последние 30 дней (нижняя граница);
  freq+pop  — товары из истории по частоте (при равенстве — по свежести), остаток — популярное;
  tifu-knn  — упрощённый TIFU-KNN: вектор пользователя = покупки с затуханием по времени,
              прогноз = α·свой вектор + (1−α)·средний вектор K ближайших соседей (косинус).
"""
import numpy as np

from metrics import topk


def history_matrix(split, users, n_items, tau=None, t_ref=None):
    """(U, N): число покупок (tau=None) или Σ exp(−(t_ref − t)/tau)."""
    M = np.zeros((len(users), n_items))
    for r, u in enumerate(users):
        it, ts = split.hist[u]
        w = np.ones(len(it)) if tau is None else np.exp(-((split.t0 if t_ref is None else t_ref) - ts) / tau)
        np.add.at(M[r], it, w)
    return M


def recent_pop_scores(split, n_items, days=30):
    ev = split.events[split.events.ts >= split.t0 - days]
    return np.bincount(ev["item"], minlength=n_items).astype(float)


def pop(split, users, n_items, k):
    s = recent_pop_scores(split, n_items)
    return np.tile(topk(s[None], k), (len(users), 1))


def freq_pop(split, users, n_items, k):
    cnt = history_matrix(split, users, n_items)
    fresh = history_matrix(split, users, n_items, tau=30.0)        # ≤ cnt → только тай-брейк
    p = recent_pop_scores(split, n_items)
    p = 0.999 * p / (p.max() + 1e-12)                              # в [0, 1): ниже любого повтора
    scores = np.where(cnt > 0, 1.0 + cnt + 1e-3 * fresh, p[None, :])
    return topk(scores, k)


def tifu_knn(split, users, n_items, k, tau=30.0, n_neighbors=100, alpha=0.7):
    pool = np.array(sorted(split.hist))                            # соседи — все, у кого есть история
    V = history_matrix(split, pool, n_items, tau=tau)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    row = {u: r for r, u in enumerate(pool)}
    Vu = V[[row[u] for u in users]]
    S = Vu @ V.T
    S[np.arange(len(users)), [row[u] for u in users]] = -np.inf     # себя не считаем соседом
    nb = np.argpartition(-S, n_neighbors, axis=1)[:, :n_neighbors]
    W = np.zeros_like(S)
    W[np.arange(len(users))[:, None], nb] = 1.0 / n_neighbors
    neigh = W @ V                                                   # среднее по соседям
    return topk(alpha * Vu + (1 - alpha) * neigh, k)


TIFU_GRID = [(tau, kn, a) for tau in (15.0, 60.0) for kn in (50, 300) for a in (0.5, 0.8)]
