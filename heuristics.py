"""
Эвристические головы (без нейросетей, только numpy). Отдают те же Components, что и нейросеть,
поэтому все правила смешивания (quota / switch / route / oracle) применяются одинаково.

  base    — одна шкала «любимые категории × популярность», купленное не исключаем;
  repeat  — насколько «пора» купить товар из истории снова (по медианному интервалу повтора);
  explore — та же шкала категорий × популярность, но без товаров из истории;
  p       — «созрел» ли самый готовый к повтору товар.
"""
import numpy as np

from mixing import Components


def item_intervals(events):
    """Медианный интервал между повторными покупками одного товара (по всем пользователям)."""
    tr = events.sort_values(["user", "item", "ts"])
    diffs = tr.groupby(["user", "item"])["ts"].diff().dropna()
    diffs = diffs[diffs > 0]
    med = tr.loc[diffs.index].assign(d=diffs).groupby("item")["d"].median()
    return med.to_dict()


def repeat_scores(h_items, h_times, t_now, intervals):
    """
    Для каждого товара из истории: насколько «пора» купить снова, в [0, 1].
    1 — с последней покупки прошло ровно типичное время; рано или сильно поздно — меньше.
    Товары, которые никто не покупал повторно (нет интервала), получают 0.
    """
    last = {}
    for i, t in zip(h_items, h_times):
        last[i] = t
    scores = {}
    for i, t_last in last.items():
        if i not in intervals:
            continue
        age = max(t_now - t_last, 1e-3)
        scores[i] = float(np.exp(-abs(np.log(age / intervals[i]))))
    return scores


def explore_scores(h_items, item_cat, item_pop, n_cats):
    """Аффинность к категориям из истории × популярность товара (сглаженно)."""
    cat_aff = np.ones(n_cats)                 # сглаживание
    np.add.at(cat_aff, item_cat[h_items], 5)
    cat_aff /= cat_aff.sum()
    return cat_aff[item_cat] * item_pop


def heur_components(split, users, items):
    """Возвращает (base_scores (U, N), Components). Статистики — только по событиям до окна."""
    item_cat = items["cat"].to_numpy()
    if (item_cat < 0).all():                  # нет категорий — одна общая
        item_cat = np.zeros_like(item_cat)
    n_cats, n_items = item_cat.max() + 1, len(items)
    item_pop = np.bincount(split.events["item"], minlength=n_items).astype(float) + 1
    item_pop /= item_pop.sum()
    intervals = item_intervals(split.events)

    U = len(users)
    base = np.zeros((U, n_items))
    rep = np.zeros((U, n_items))
    exp = np.zeros((U, n_items))
    p = np.zeros(U)
    for r, u in enumerate(users):
        h_items, h_times = split.hist[u]
        e = explore_scores(h_items, item_cat, item_pop, n_cats)
        base[r] = e
        e = e.copy()
        e[np.unique(h_items)] = 0
        exp[r] = e / e.sum()
        sc = repeat_scores(h_items, h_times, split.t_query[u], intervals)
        if sc:
            idx = np.fromiter(sc.keys(), dtype=np.int64)
            val = np.fromiter(sc.values(), dtype=float)
            rep[r, idx] = val / val.sum()
            p[r] = val.max()
    return base, Components(rep, exp, p)
