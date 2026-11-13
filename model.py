"""
Одностадийная генеративная модель: трансформер над историей покупок → распределение над всем каталогом.

Режимы:
  base   — одна голова: softmax над всеми товарами (как обычная одностадийная модель);
  switch — две головы + переключатель (идея RepeatNet):
             repeat-голова  — указатель (pointer) на товары из истории, учитывает «возраст» покупки;
             explore-голова — softmax только по товарам, которых нет в истории;
             p = sigmoid(w·h) — вероятность режима повторов на этот запрос;
             P(item) = p · P_repeat(item) + (1 − p) · P_explore(item).
  fixed30 использует ту же обученную switch-модель, но смешивает головы жёсткой квотой при выдаче.

Борьба с «потраченными слотами» (повтор уже купленного холодильника), опции repeat-головы:
  rep_bias  — обучаемый скаляр «повторяемости» на каждый товар, добавляется к логиту указателя;
  rep_prior — логит сглаженной доли повторных покупок товара из train (счётчик, не обучается)
              с обучаемым весом; обобщается на редкие товары лучше, чем rep_bias;
  rep_null  — «пустой» слот указателя: если в истории нечего повторять, масса уходит в него
              и перетекает в explore. Указатель обязан распределить 1.0 по истории, и без null-слота
              даже нерелевантные товары истории получают заметную массу — а на выдаче
              p·P_repeat легко обгоняет (1−p)·P_explore, размазанную по всему каталогу.
              С null-слотом: P = p·P_rep + (1 − p + p·p_null)·P_explore, эффективный переключатель p·(1 − p_null).
Без опций модель совпадает с исходной.

Индексы товаров в модели: 1..n_items, 0 — паддинг. Паддинг справа (последовательность начинается с 0-й позиции).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BUCKETS = 32
NEG = -1e9


def time_bucket(dt_days):
    """Лог-корзины по времени: ~час, несколько часов, день, неделя, месяц... (до ~года)."""
    hours = torch.clamp(dt_days, min=0) * 24.0
    b = torch.floor(torch.log2(hours + 1.0) * 2).long()
    return torch.clamp(b, 0, N_BUCKETS - 1)


class OneStageRec(nn.Module):
    def __init__(self, n_items, mode="switch", d=64, n_layers=2, n_heads=2, max_len=50, dropout=0.1,
                 rep_bias=False, rep_null=False, rep_prior=None):
        super().__init__()
        assert mode in ("base", "switch")
        self.mode = mode
        self.n_items = n_items
        self.d = d
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        self.gap_emb = nn.Embedding(N_BUCKETS, d)        # пауза между соседними покупками
        self.query_emb = nn.Embedding(N_BUCKETS, d)      # сколько прошло от последней покупки до запроса
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)
        if mode == "switch":
            self.age_emb = nn.Embedding(N_BUCKETS, d)    # «возраст» покупки из истории на момент запроса
            self.rep_query = nn.Linear(d, d)
            self.switch = nn.Linear(d, 1)
            self.rep_bias = nn.Embedding(n_items + 1, 1) if rep_bias else None
            self.null_key = nn.Parameter(torch.zeros(d)) if rep_null else None
            if rep_prior is not None:          # (n_items + 1,), индекс 0 — паддинг
                self.register_buffer("rep_prior", torch.as_tensor(rep_prior, dtype=torch.float32))
                self.prior_w = nn.Parameter(torch.tensor(1.0))
            else:
                self.rep_prior = None
        # Эмбеддинги N(0,1) при d=64 дают логиты со std ~10 на старте (softmax насыщен).
        # Стандартная инициализация SASRec/BERT: std=0.02.
        for emb in self.modules():
            if isinstance(emb, nn.Embedding):
                nn.init.normal_(emb.weight, std=0.02)
                if emb.padding_idx is not None:
                    with torch.no_grad():
                        emb.weight[emb.padding_idx].zero_()
        if mode == "switch" and self.rep_bias is not None:
            nn.init.zeros_(self.rep_bias.weight)     # старт = исходная модель

    def encode(self, items, times, query_times):
        """
        items, times:  (B, L) — история; query_times: (B, L) — момент запроса для каждой позиции
        (в обучении это время следующей покупки, в инференсе — «сейчас»).
        Возвращает q: (B, L, d) — состояние пользователя к моменту запроса.
        """
        B, L = items.shape
        prev = torch.cat([times[:, :1], times[:, :-1]], dim=1)
        pos = torch.arange(L, device=items.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(items) + self.pos_emb(pos) + self.gap_emb(time_bucket(times - prev))
        x = self.dropout(x)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=items.device), diagonal=1)
        h = self.norm(self.encoder(x, mask=causal))
        return h + self.query_emb(time_bucket(query_times - times))

    def forward(self, items, times, query_times, pre_seen=None):
        """
        base:   возвращает (log_probs, None, None, None)
        switch: возвращает (log_probs смеси, p_repeat (нормированная), p_explore, p_switch (эффективный))
        log_probs/p_*: (B, L, n_items + 1), p_switch: (B, L, 1)
        pre_seen: (B, n_items + 1) bool — товары, купленные ДО окна последних max_len покупок.
                  Explore-голова по определению — «чего не было в истории», поэтому исключает и их;
                  без этого холодильник, выпавший из окна, рекомендуется заново.
        """
        q = self.encode(items, times, query_times)
        logits = q @ self.item_emb.weight.T                       # (B, L, N+1)
        pad_col = torch.zeros_like(logits, dtype=torch.bool)
        pad_col[..., 0] = True
        logits = logits.masked_fill(pad_col, NEG)

        if self.mode == "base":
            return F.log_softmax(logits, dim=-1), None, None, None

        B, L = items.shape
        # --- repeat-голова: внимание позиции i на покупки j <= i с учётом их возраста к моменту запроса
        age = query_times.unsqueeze(2) - times.unsqueeze(1)       # (B, L_i, L_j)
        # att_ij = rq_i · (e_j + age_emb[bucket_ij]) = rq_i·e_j + (rq_i·age_emb)[bucket_ij]
        # — без тензора (B, L, L, d): вторая часть берётся gather'ом из (B, L, N_BUCKETS)
        rq = self.rep_query(q)                                                     # (B, L, d)
        att_item = rq @ self.item_emb(items).transpose(1, 2)                      # (B, L, L)
        att_age = (rq @ self.age_emb.weight.T).gather(2, time_bucket(age))        # (B, L, L)
        att = (att_item + att_age) / math.sqrt(self.d)
        if self.rep_bias is not None:                              # «повторяемость» товара j
            att = att + self.rep_bias(items).squeeze(-1).unsqueeze(1)
        if self.rep_prior is not None:
            att = att + self.prior_w * self.rep_prior[items].unsqueeze(1)
        lower = torch.tril(torch.ones(L, L, dtype=torch.bool, device=items.device))
        valid = lower.unsqueeze(0) & (items != 0).unsqueeze(1)    # (B, L_i, L_j)
        att = att.masked_fill(~valid, NEG)
        if self.null_key is not None:
            null = (rq @ self.null_key / math.sqrt(self.d)).unsqueeze(-1)            # (B, L, 1)
            w = torch.softmax(torch.cat([att, null], dim=-1), dim=-1)
            w, p_null = w[..., :L], w[..., L:]
        else:
            w = torch.softmax(att, dim=-1)
            p_null = torch.zeros_like(att[..., :1])

        # складываем веса позиций в вероятности товаров (один товар мог встречаться несколько раз)
        index = items.unsqueeze(1).expand(B, L, L).contiguous()
        p_rep = torch.zeros_like(logits).scatter_add(2, index, w)
        seen = torch.zeros_like(logits).scatter_add(2, index, valid.float()) > 0
        if pre_seen is not None:
            seen = seen | pre_seen.unsqueeze(1)

        # --- explore-голова: только товары, которых нет в истории
        p_exp = torch.softmax(logits.masked_fill(seen, NEG), dim=-1)

        # --- переключатель
        p_sw = torch.sigmoid(self.switch(q))                     # (B, L, 1)
        probs = p_sw * p_rep + (1 - p_sw + p_sw * p_null) * p_exp
        # для правил смешивания: нормированная repeat-голова и эффективный переключатель,
        # p_eff·rep_norm + (1 − p_eff)·exp == probs
        p_eff = p_sw * (1 - p_null)
        rep_norm = p_rep / (1 - p_null).clamp_min(1e-9)
        return torch.log(probs + 1e-10), rep_norm, p_exp, p_eff
