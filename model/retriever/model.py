"""Two-Tower architecture.

ItemTower: 9 item features -> concat -> MLP -> L2-norm -> [*, d].
UserTower: pooled history (mean of item vecs; empty -> learned h_empty) + gender + joined
           -> MLP -> L2-norm -> [B, d].
TwoTower: both towers plus the item-vec cache read by the history branch (refreshed often, detached).

No user-id embedding: eval holds out whole users, so their ids would never be trained.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers = []
    prev = in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _masked_mean(vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """vecs [*, L, D], mask [*, L] bool -> mean over the unmasked L positions. Empty -> 0 (caller handles it)."""
    m = mask.unsqueeze(-1).to(vecs.dtype)
    summed = (vecs * m).sum(dim=-2)
    cnt = m.sum(dim=-2).clamp(min=1.0)
    return summed / cnt


def _masked_weighted_mean(vecs: torch.Tensor, mask: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Like _masked_mean with a weight w [*, L] > 0 per position, so the denominator stays > 0.
    Empty -> 0 (caller handles it)."""
    m = (mask.to(vecs.dtype) * w.to(vecs.dtype)).unsqueeze(-1)   # [*, L, 1]
    summed = (vecs * m).sum(dim=-2)
    denom = m.sum(dim=-2).clamp(min=1e-6)
    return summed / denom


class ItemTower(nn.Module):
    """Item features -> d-dim vector (L2-normalized)."""

    CAT_FEATS = [
        ("type", "type_id"), ("source", "source_id"), ("rating", "rating_id"),
        ("demographics", "demographics_id"), ("start_year", "startyear_bucket"),
        ("episodes", "episodes_bucket"),
    ]

    def __init__(self, spec: dict, d: int, genres_proj: int, themes_proj: int, hidden: List[int],
                 num_items: int, id_dim: int, use_item_id: bool,
                 raw_synopsis_dim: int = None, use_synopsis: bool = False, synopsis_dim: int = 48,
                 synopsis_proj_hidden: List[int] = None, synopsis_normalize: str = "none"):
        super().__init__()
        item = spec["item_features"]
        self.cat_emb = nn.ModuleDict()
        cat_dim = 0
        for name, _col in self.CAT_FEATS:
            s = item[name]
            self.cat_emb[f"cat_{name}"] = nn.Embedding(s["vocab"], s["dim"], padding_idx=0)
            cat_dim += s["dim"]

        gw = item["genres"]["width"]
        tw = item["themes"]["width"]
        self.genres_proj = nn.Linear(gw, genres_proj)
        self.themes_proj = nn.Linear(tw, themes_proj)

        st = item["studios"]
        # No padding_idx: emb[0] is the learned "no studio" token (empty_id = 0). Structural
        # padding is removed by the mask in forward instead.
        self.studio_emb = nn.Embedding(st["vocab"], st["dim"])
        studio_dim = st["dim"]

        # anime-id embedding (collaborative residual): PAD = 0 stays zero, OOV (1) and real ids learn.
        self.use_item_id = use_item_id
        in_dim = cat_dim + genres_proj + themes_proj + studio_dim
        if use_item_id:
            self.id_emb = nn.Embedding(num_items, id_dim, padding_idx=0)
            in_dim += id_dim

        # Synopsis: project the frozen raw_synopsis_dim vector to synopsis_dim, into the content
        # path. Low-info rows (missing/short/placeholder text) use the learned no_synopsis vector,
        # which avoids NaN from normalizing a ~0 vector; PAD/OOV are low-info too.
        self.use_synopsis = use_synopsis and raw_synopsis_dim is not None
        self.synopsis_normalize = synopsis_normalize
        if self.use_synopsis:
            self.synopsis_proj = _mlp(raw_synopsis_dim, synopsis_proj_hidden or [], synopsis_dim)
            self.no_synopsis = nn.Parameter(torch.zeros(synopsis_dim))
            nn.init.normal_(self.no_synopsis, std=0.02)
            in_dim += synopsis_dim
        self.mlp = _mlp(in_dim, hidden, d)

    def forward(self, items: "ItemBatch", id_idx: torch.Tensor) -> torch.Tensor:
        parts = [self.cat_emb[f"cat_{name}"](items.cat[col]) for name, col in self.CAT_FEATS]
        parts.append(self.genres_proj(items.genres))
        parts.append(self.themes_proj(items.themes))
        # Studios: average-pooled embeddings, where id 0 is both padding and empty_id.
        # Invariant from data_prep/01_item_features.py: id 0 only ever appears as the standalone
        # list [0]. So an all-zero row is an item without studios (use emb[0]); a 0 inside a
        # non-empty row is padding (masked).
        st_vec = self.studio_emb(items.studios)                 # [*, S, dim]
        st_mask = items.studios != 0                            # [*, S] pad and empty are both 0
        empty = ~st_mask.any(dim=-1)                            # [*] all-zero row = no studios
        st_mask[..., 0] = st_mask[..., 0] | empty               # empty -> emb[0]
        parts.append(_masked_mean(st_vec, st_mask))
        if self.use_synopsis:
            raw = items.synopsis
            if self.synopsis_normalize == "l2":
                raw = F.normalize(raw, dim=-1)                  # 'none' = artifact already L2 (default)
            proj = self.synopsis_proj(raw)
            low = items.synopsis_low_info.unsqueeze(-1)         # [*,1] -> broadcast
            parts.append(torch.where(low, self.no_synopsis, proj))
        if self.use_item_id:
            parts.append(self.id_emb(id_idx))                   # id_idx may be dropped out to OOV
        x = torch.cat(parts, dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    """Pooled history + gender + joined -> MLP -> d-dim vector (L2-normalized)."""

    def __init__(self, spec: dict, d: int, hidden: List[int]):
        super().__init__()
        uf = spec["user_features"]
        self.gender_emb = nn.Embedding(uf["gender"]["vocab"], uf["gender"]["dim"], padding_idx=0)
        self.joined_emb = nn.Embedding(uf["joined"]["vocab"], uf["joined"]["dim"])
        self.h_empty = nn.Parameter(torch.zeros(d))             # learned, replaces pooling when empty
        nn.init.normal_(self.h_empty, std=0.02)

        in_dim = d + uf["gender"]["dim"] + uf["joined"]["dim"]
        self.mlp = _mlp(in_dim, hidden, d)

    def forward(self, hist_pooled, gender_id, joined_bucket) -> torch.Tensor:
        x = torch.cat([hist_pooled, self.gender_emb(gender_id), self.joined_emb(joined_bucket)], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class ItemBatch:
    """Feature tensors of one batch of anime_idx (gathered from ItemTable) for ItemTower.forward."""

    def __init__(self, cat: Dict[str, torch.Tensor], genres, themes, studios,
                 synopsis=None, synopsis_low_info=None):
        self.cat = cat
        self.genres = genres
        self.themes = themes
        self.studios = studios
        self.synopsis = synopsis                 # [*, raw_dim], None when synopsis is off
        self.synopsis_low_info = synopsis_low_info  # [*] bool or None


class TwoTower(nn.Module):
    def __init__(self, spec: dict, cfg, item_table):
        super().__init__()
        self.d = cfg.d
        self.use_item_id = cfg.use_item_id
        self.id_dropout = cfg.id_dropout
        self.oov_idx = spec["special_idx"]["OOV"]
        # raw synopsis dim comes from the artifact (ItemTable.synopsis_emb); no artifact -> branch off
        raw_syn = getattr(item_table, "synopsis_emb", None)
        raw_synopsis_dim = raw_syn.shape[1] if (cfg.use_synopsis and raw_syn is not None) else None
        if cfg.use_synopsis and raw_syn is None:
            print("WARNING: use_synopsis=True but ItemTable has no synopsis_emb (missing artifact?) "
                  "-> synopsis skipped")
        self.item_tower = ItemTower(spec, cfg.d, cfg.genres_proj, cfg.themes_proj, cfg.mlp_hidden,
                                    item_table.num_items, cfg.id_dim, cfg.use_item_id,
                                    raw_synopsis_dim=raw_synopsis_dim, use_synopsis=cfg.use_synopsis,
                                    synopsis_dim=cfg.synopsis_dim, synopsis_proj_hidden=cfg.synopsis_proj_hidden,
                                    synopsis_normalize=cfg.synopsis_normalize)
        self.user_tower = UserTower(spec, cfg.d, cfg.mlp_hidden)
        # Score-weighted history pooling: none|linear|learned. learned = a positive weight per
        # score level 0..10 (11 buckets); padding is masked, so no padding_idx is needed.
        self.score_pool = cfg.score_pool
        if cfg.score_pool == "learned":
            self.score_weight = nn.Embedding(11, 1)
        # Attention pooling (history_pool='attn'): a learned query attends over the item vecs
        # (values = item vecs), a content-aware convex combination instead of a uniform mean.
        # Ignores score_pool; item_cache stays detached.
        self.history_pool = cfg.history_pool
        if cfg.history_pool == "attn":
            self.attn_key = nn.Linear(self.d, self.d, bias=False)
            self.attn_query = nn.Parameter(torch.randn(self.d))   # unit scale: with /sqrt(d) logits start unit-variance
        # History-side vectors: 'cache' (detached item vecs, no grad) | 'embed' (own trainable
        # table, gradients flow through the history path; export.py ships it as hist_emb).
        self.history_source = cfg.history_source
        if cfg.history_source == "embed":
            self.hist_emb = nn.Embedding(item_table.num_items, cfg.d, padding_idx=0)
        self.item_table = item_table                            # tensors on the model's device
        self.register_buffer("item_cache", torch.zeros(item_table.num_items, cfg.d), persistent=False)

    # --- item side ---
    def _gather(self, idx: torch.Tensor) -> ItemBatch:
        it = self.item_table
        cat = {col: it.cat[col][idx] for col in it.CAT_COLS}
        syn = getattr(it, "synopsis_emb", None)
        if syn is not None:
            return ItemBatch(cat, it.genres[idx], it.themes[idx], it.studios[idx],
                             syn[idx], it.synopsis_low_info[idx])
        return ItemBatch(cat, it.genres[idx], it.themes[idx], it.studios[idx])

    def encode_items(self, idx: torch.Tensor) -> torch.Tensor:
        """Encode a batch of anime_idx of any shape -> [*, d], with grad. In training, id-dropout
        masks ids to OOV (content stays real) to simulate new anime and teach the backoff."""
        shape = idx.shape
        flat = idx.reshape(-1)
        id_flat = flat
        if self.use_item_id and self.training and self.id_dropout > 0:
            drop = (torch.rand_like(flat, dtype=torch.float) < self.id_dropout) & (flat >= 2)
            id_flat = torch.where(drop, torch.full_like(flat, self.oov_idx), flat)
        vec = self.item_tower(self._gather(flat), id_flat)
        return vec.reshape(*shape, self.d)

    @torch.no_grad()
    def refresh_item_cache(self, chunk: int = 8192, cold_mask: torch.Tensor = None):
        """cold_mask [num_items] bool: True = cold item, its id branch is encoded as OOV
        (simulates an out-of-vocab anime at serve time; content stays real). None = warm (real ids)."""
        self.eval()
        N = self.item_table.num_items
        if cold_mask is not None:
            cold_mask = cold_mask.to(self.item_cache.device)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            idx = torch.arange(s, e, device=self.item_cache.device)
            id_idx = idx
            if cold_mask is not None:
                id_idx = torch.where(cold_mask[s:e], torch.full_like(idx, self.oov_idx), idx)
            self.item_cache[s:e] = self.item_tower(self._gather(idx), id_idx)
        self.train()

    # --- user side ---
    def _attn_pool(self, vecs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """A learned query attends over the history vecs (content reweighting, values = vecs).
        An empty row gets logits 0 (uniform; softmax over all -inf is NaN) and the caller
        overwrites it with h_empty."""
        logits = (self.attn_key(vecs) @ self.attn_query) / self.d ** 0.5   # [B, L]
        logits = logits.masked_fill(~mask, float("-inf"))
        logits = logits.masked_fill(~mask.any(dim=-1, keepdim=True), 0.0)  # empty row -> uniform (NaN-safe)
        attn = torch.softmax(logits, dim=-1)                               # [B, L]
        return (attn.unsqueeze(-1) * vecs).sum(dim=-2)                     # [B, d]

    def pool_history(self, history_ids, history_mask, history_scores=None) -> torch.Tensor:
        """Pool the history vectors. Source = history_source: 'cache' (detached item vecs, no grad)
        | 'embed' (trainable table, with grad). An empty row -> h_empty. history_pool: mean
        (weighted by score_pool none/linear/learned) | attn (learned-query attention, ignores scores)."""
        if self.history_source == "embed":
            vecs = self.hist_emb(history_ids)                  # [B, L, d] (grad)
        else:
            vecs = self.item_cache[history_ids].detach()       # [B, L, d]
        if self.history_pool == "attn":
            pooled = self._attn_pool(vecs, history_mask)       # [B, d]
        elif self.score_pool == "none" or history_scores is None:
            pooled = _masked_mean(vecs, history_mask)          # [B, d]
        else:
            if self.score_pool == "learned":
                w = F.softplus(self.score_weight(history_scores).squeeze(-1))   # [B, L] >0
            else:                                              # linear: the score itself
                w = history_scores.to(vecs.dtype).clamp(min=1.0)
            pooled = _masked_weighted_mean(vecs, history_mask, w)
        empty = ~history_mask.any(dim=-1)                       # [B]
        pooled = torch.where(empty.unsqueeze(-1), self.user_tower.h_empty.to(pooled.dtype), pooled)
        return pooled

    def encode_users(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pooled = self.pool_history(batch["history_ids"], batch["history_mask"], batch.get("history_scores"))
        return self.user_tower(pooled, batch["gender_id"], batch["joined_bucket"])

    def forward(self, batch: Dict[str, torch.Tensor]):
        U = self.encode_users(batch)                           # [B, d]
        V_pos = self.encode_items(batch["pos"])                # [B, d] (grad)
        V_hn = self.encode_items(batch["hardneg_ids"])         # [B, m, d] (grad)
        return U, V_pos, V_hn
