"""
Данные и протокол оценки для эксперимента «одностадийная модель + режим повторов».

  make_synthetic()        — синтетический маркетплейс с известной правдой (какие категории расходные),
                            регулятором доли повторов и шумом;
  make_protocol()         — сплит по времени: train / val-окно / test-окно, цели и время запроса;
  load_tecd_marketplace() — загрузчик T-ECD (в экспериментах сейчас не используется).

Формат:
  events: DataFrame[user, item, ts]   ts — время в днях (float), user/item — int с 0
  items:  DataFrame[item, cat, is_consumable]
"""
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


def make_synthetic(n_users=4000, n_items=1500, n_cats=40, n_personas=8,
                   consumable_share=0.4, durable_rate=0.5, days=365, brand_switch=0.03,
                   pause_prob=0.3, seasonal_share=0.3, seed=0):
    """
    Регуляторы доли повторов среди целей:
      consumable_share — доля расходных категорий;
      durable_rate     — сколько ещё (сверх одного) товаров длительного пользования в среднем
                         покупается за год в каждой любимой категории (Пуассон).

    Шум, чтобы данные не повторяли устройство модели один в один:
      brand_switch   — на каждой покупке расходника шанс насовсем сменить любимый товар
                       (повтор на уровне категории, но товар новый → explore-цель);
      pause_prob     — доля пользователей с паузой 2–8 недель (отпуск): покупок нет, а после
                       возвращения запас кончился — расходник покупается сразу, «не по графику»;
      seasonal_share — доля категорий длительного пользования с сезонным пиком спроса.
    """
    rng = np.random.default_rng(seed)

    # --- категории: расходные (корм, вода) или длительного пользования (холодильник)
    cat_is_cons = rng.random(n_cats) < consumable_share
    cat_interval = rng.uniform(7, 45, n_cats)          # типичный период повторной покупки, дни
    seasonal = (~cat_is_cons) & (rng.random(n_cats) < seasonal_share)
    cat_peak = rng.uniform(0, days, n_cats)            # день пика для сезонных категорий

    # --- товары: категория + популярность (тяжёлый хвост)
    item_cat = rng.integers(0, n_cats, n_items)
    item_pop = np.minimum(rng.zipf(1.6, n_items), 1000).astype(float)
    items_by_cat = [np.where(item_cat == c)[0] for c in range(n_cats)]

    # --- «персоны»: группы похожих пользователей, чтобы новое было предсказуемо коллаборативно
    persona_w = rng.dirichlet(np.full(n_cats, 0.3), n_personas)

    rows = []
    for u in range(n_users):
        persona = rng.integers(n_personas)
        n_pref = rng.integers(6, 14)
        pref = rng.choice(n_cats, n_pref, replace=False, p=persona_w[persona])

        pause = None
        if rng.random() < pause_prob:
            s = rng.uniform(0, days)
            pause = (s, s + rng.uniform(14, 60))

        def in_pause(t):
            return pause is not None and pause[0] <= t < pause[1]

        for c in pref:
            cand = items_by_cat[c]
            if len(cand) == 0:
                continue
            p = item_pop[cand] / item_pop[cand].sum()

            if cat_is_cons[c]:
                # расходник: любимый товар + личный период покупки
                fav = rng.choice(cand, size=min(2, len(cand)), replace=False, p=p)
                interval = cat_interval[c] * rng.uniform(0.6, 1.6)
                t = rng.uniform(0, interval)
                while t < days:
                    if in_pause(t):
                        t = pause[1] + rng.uniform(0, 3)       # вернулся, запас кончился
                        continue
                    if rng.random() < brand_switch:           # сменил бренд насовсем
                        fav[0] = rng.choice(cand, p=p)
                    item = fav[0]
                    if len(fav) > 1 and rng.random() < 0.2:
                        item = fav[1]
                    if rng.random() < 0.1:                    # иногда пробует новое в категории
                        item = rng.choice(cand, p=p)
                    rows.append((u, int(item), t))
                    t += interval * rng.lognormal(0, 0.25)
            else:
                # длительного пользования: 1 + Poisson(durable_rate) разных товаров за год, без повторов
                bought = set()
                for _ in range(1 + rng.poisson(durable_rate)):
                    item = int(rng.choice(cand, p=p))
                    t = rng.normal(cat_peak[c], 15) % days if seasonal[c] else rng.uniform(0, days)
                    if item in bought or in_pause(t):
                        continue
                    bought.add(item)
                    rows.append((u, item, t))

    events = pd.DataFrame(rows, columns=["user", "item", "ts"])
    events = events.sort_values(["user", "ts"]).reset_index(drop=True)
    items = pd.DataFrame({"item": np.arange(n_items), "cat": item_cat,
                          "is_consumable": cat_is_cons[item_cat]})
    return events, items


# ----------------------------------------------------------------------------- протокол

@dataclass
class Split:
    hist: dict            # user -> (items, times): история до начала окна, по времени
    targets: dict         # user -> np.array различных целевых товаров
    t_query: dict         # user -> время запроса
    events: pd.DataFrame  # все события до начала окна (для статистик: популярность, интервалы)
    t0: float             # начало окна

    @property
    def users(self):
        return np.array(sorted(self.targets))


def user_histories(events):
    """user -> (массив товаров, массив времён), по возрастанию времени."""
    return {u: (g["item"].to_numpy(), g["ts"].to_numpy()) for u, g in events.groupby("user")}


def make_protocol(events, protocol="window", t_val=351.0, t_test=358.0, window=7.0, gap=0.0):
    """
    Протоколы:
      window     — (по умолчанию) запрос в момент t0, цели — все различные товары, купленные
                   в [t0+gap, t0+gap+window). Когда именно человек купит, модели неизвестно.
      next       — старый протокол: одна цель (первая покупка после t0), время запроса = время
                   этой покупки. Утечка: модель знает момент покупки.
      next-start — одна цель, но время запроса = t0. Разница next и next-start = цена утечки.

    Модель учится на событиях < t_val. Val-окно — для ранней остановки и подбора квот/порогов.
    На тесте модель НЕ переобучается, но история пользователя дополняется событиями до t_test.
    """
    assert protocol in ("window", "next", "next-start")
    assert t_val + gap + window <= t_test + 1e-9, "val-окно не должно залезать в тест"

    def build(t0):
        before = events[events.ts < t0]
        hist = user_histories(before)
        w = events[(events.ts >= t0 + gap) & (events.ts < t0 + gap + window)]
        w = w[w.user.isin(list(hist))]                 # нужна хоть какая-то история
        targets, t_query = {}, {}
        if protocol == "window":
            for u, g in w.groupby("user"):
                targets[u] = np.unique(g["item"].to_numpy())
                t_query[u] = t0
        else:
            first = w.sort_values("ts").groupby("user").head(1)
            for u, i, ts in zip(first.user, first.item, first.ts):
                targets[u] = np.array([i])
                t_query[u] = ts if protocol == "next" else t0
        return Split(hist, targets, t_query, before.reset_index(drop=True), t0)

    train = events[events.ts < t_val].reset_index(drop=True)
    return train, build(t_val), build(t_test)


# ----------------------------------------------------------------------------- T-ECD (не используется)

def load_tecd_marketplace(root, user_col="user_id", item_col="item_id", ts_col="timestamp",
                          action_col="action_type", keep_actions=None, max_files=None,
                          top_items=50000, min_events=3):
    """
    Читает marketplace/events/{day}.pq из T-ECD (нужен pyarrow).
    Имена колонок и тип «покупочного» действия проверь в карточке датасета на HF
    (t-tech/T-ECD) и передай аргументами — в статье на Хабре они не указаны.
    keep_actions — список действий, которые считаем «потреблением» (например, переход к покупке).
    top_items    — оставляем самые частые товары: полный softmax по 2.6 млн товаров не влезет.
    """
    files = sorted(glob.glob(os.path.join(root, "marketplace", "events", "*.pq")))
    if max_files:
        files = files[:max_files]
    parts = []
    for f in files:
        df = pd.read_parquet(f)
        if keep_actions is not None:
            df = df[df[action_col].isin(keep_actions)]
        parts.append(df[[user_col, item_col, ts_col]])
    df = pd.concat(parts, ignore_index=True)
    df.columns = ["user_raw", "item_raw", "ts_raw"]

    # время → дни от начала
    if np.issubdtype(df["ts_raw"].dtype, np.datetime64):
        ts = (df["ts_raw"] - df["ts_raw"].min()).dt.total_seconds() / 86400
    else:
        ts = (df["ts_raw"] - df["ts_raw"].min()) / 86400        # если секунды
    df["ts"] = ts.astype(float)

    keep = df["item_raw"].value_counts().index[:top_items]
    df = df[df["item_raw"].isin(keep)]
    n_per_user = df.groupby("user_raw")["item_raw"].transform("size")
    df = df[n_per_user >= min_events].copy()

    df["user"] = df["user_raw"].astype("category").cat.codes
    df["item"] = df["item_raw"].astype("category").cat.codes
    events = df[["user", "item", "ts"]].sort_values(["user", "ts"]).reset_index(drop=True)

    n_items = events["item"].max() + 1
    items = pd.DataFrame({"item": np.arange(n_items), "cat": -1, "is_consumable": np.nan})
    return events, items
