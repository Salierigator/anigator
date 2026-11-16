"""Config for the Two-Tower retriever.

Every tunable hyperparameter lives in TwoTowerConfig: presets and sweeps in `runs.py`
override the defaults and pass the result to data/model/train. Feature vocab sizes and
dims are not hard-coded here; they are read from feature_spec.json.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from model.paths import TRAIN

TRAIN_DATA = TRAIN / "retriever" / "data"
RUNS = TRAIN / "retriever" / "runs"


def auto_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TwoTowerConfig:
    # --- model ---
    d: int = 128                      # shared embedding dim (output of both towers)
    genres_proj: int = 8             # Linear(22 -> genres_proj)
    themes_proj: int = 8             # Linear(53 -> themes_proj)
    mlp_hidden: List[int] = field(default_factory=lambda: [256])
    use_item_id: bool = False        # anime-id embedding in ItemTower
    id_dim: int = 64                 # id-embedding dim (table num_items x id_dim)
    id_dropout: float = 0.2          # per-item prob of masking the id to OOV in training (content backoff)
    score_pool: str = "none"         # score-weighted history pooling: 'none'|'linear'|'learned'
    history_pool: str = "mean"       # 'mean' (+ score_pool) | 'attn' (learned-query attention, ignores score_pool)
    history_source: str = "cache"    # history-side vectors: 'cache' (detached item vecs) | 'embed' (own trainable table)

    # --- synopsis (frozen text embedding per item + trainable projection) ---
    use_synopsis: bool = False       # synopsis branch in ItemTower
    synopsis_dim: int = 48           # dim after projection, concatenated into the content path
    synopsis_proj_hidden: List[int] = field(default_factory=list)  # hidden layers of the raw->dim MLP; [] = one Linear
    synopsis_normalize: str = "none" # frozen vecs before projection: 'none' (artifact is already L2) | 'l2'
    synopsis_emb_file: str = "synopsis_emb.npy"            # frozen [num_items, raw_dim]
    synopsis_low_info_file: str = "synopsis_low_info.npy"  # bool [num_items]: missing/short synopsis -> learned vec

    # --- loss ---
    tau: float = 0.07                # temperature
    beta: float = 1.0                # weight of the hard-neg branch in the denominator
    logq_alpha: float = 1.0          # logQ correction strength (1.0 = full, 0 = off)

    # --- train ---
    lr: float = 1e-3
    optimizer: str = "adam"          # 'adam' | 'adamw' (differs only when weight_decay > 0)
    cosine_lr: bool = False          # cosine-anneal lr to 0 over the run instead of a constant lr
    weight_decay: float = 0.0
    batch_size: int = 4096
    epochs: int = 1
    hist_dropout: float = 0.12       # prob of dropping the whole history -> h_empty
    m_hardneg: int = 3               # hard negatives sampled per anchor
    train_hist_len: int = 32         # history items sampled per anchor from the full list (augmentation)
    max_examples_per_user: Optional[int] = None  # cap per user per epoch, resampled each epoch (None = off)
    cache_refresh_steps: int = 300   # refresh the item-vec cache every N steps
    log_every: int = 50

    # --- eval (cold-user split) ---
    eval_ks: List[int] = field(default_factory=lambda: [10, 50, 100, 200, 500])
    headline_k: int = 200            # checkpoint selection = warm val recall@headline_k
    eval_history_cap: int = 1024     # history prefix (sorted by score desc) used at eval
    eval_split: str = "val"
    eval_every_steps: int = 0        # also eval every N steps within an epoch (0 = end of epoch only)
    eval_cold_in_loop: bool = False  # do_eval also scores cold val into history (never used to pick the checkpoint)
    early_stop_patience: Optional[int] = None  # None = no early stopping
    early_stop_min_delta: float = 0.0          # minimum gain that resets patience

    # --- infra ---
    device: str = field(default_factory=auto_device)
    num_workers: int = 0
    seed: int = 42
    subset: Optional[int] = None     # smoke: keep only the first N train examples (None = all)
    train_user_frac: Optional[float] = None  # HP search: keep a random fraction of train users; catalog, logQ, eval stay full
    subset_seed: int = 12345         # seed of the train_user_frac user filter
    ckpt_dir: Path = field(default_factory=lambda: RUNS)

    # --- paths ---
    train_data: Path = TRAIN_DATA
