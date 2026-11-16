"""Model invariants: h_empty, attn NaN-safe, studios empty-vs-pad, id-dropout,
cold_oov cache, history_source cache (no grad) vs embed (grad)."""
import numpy as np
import torch

from model.twotower.retriever.core import data as data_mod


def _user_batch(B=2, L=3, empty=False):
    return {
        "history_ids": torch.zeros(B, L, dtype=torch.long) if empty
        else torch.tensor([[2, 3, 0], [4, 0, 0]]),
        "history_mask": torch.zeros(B, L, dtype=torch.bool) if empty
        else torch.tensor([[True, True, False], [True, False, False]]),
        "history_scores": torch.zeros(B, L, dtype=torch.long),
        "gender_id": torch.tensor([1, 2]),
        "joined_bucket": torch.tensor([0, 3]),
    }


def test_empty_history_h_empty_all_pools(make_model):
    for pool in ["mean", "attn"]:
        model, cfg, _ = make_model(history_pool=pool)
        model.refresh_item_cache()
        pooled = model.pool_history(torch.zeros(2, 3, dtype=torch.long),
                                    torch.zeros(2, 3, dtype=torch.bool))
        assert torch.isfinite(pooled).all(), f"{pool}: NaN with an empty history"
        assert torch.allclose(pooled[0], model.user_tower.h_empty), f"{pool}: an empty row must be h_empty"
        U = model.encode_users(_user_batch(empty=True))
        assert torch.isfinite(U).all()


def test_studios_empty_vs_pad(make_model):
    """An all-zero studios row is empty and uses emb[0] (changing emb[0] changes the output);
    a 0 at the end of a non-empty row is padding and masked (changing emb[0] does not)."""
    model, _, _ = make_model()
    model.eval()
    idx_empty, idx_nonempty = torch.tensor([4]), torch.tensor([3])  # studios [0,0] vs [4,0]
    with torch.no_grad():
        before_e = model.encode_items(idx_empty).clone()
        before_n = model.encode_items(idx_nonempty).clone()
        model.item_tower.studio_emb.weight[0] += 5.0
        after_e = model.encode_items(idx_empty)
        after_n = model.encode_items(idx_nonempty)
    assert not torch.allclose(before_e, after_e), "an item without studios must use the learned emb[0]"
    assert torch.allclose(before_n, after_n), "pad 0 in a non-empty row must be masked"


def test_id_dropout_only_train_and_real(make_model):
    model, _, _ = make_model(use_item_id=True, id_dropout=1.0)
    idx = torch.tensor([5, 6])
    model.eval()                                  # eval: real ids
    with torch.no_grad():
        v_eval = model.encode_items(idx).clone()
    model.train()                                 # train + p=1: every real id -> OOV
    with torch.no_grad():
        v_drop = model.encode_items(idx)
        v_oov = model.item_tower(model._gather(idx), torch.full_like(idx, model.oov_idx))
    assert torch.allclose(v_drop, v_oov), "id_dropout=1 in training must encode with OOV"
    assert not torch.allclose(v_eval, v_drop), "eval must use the real ids (unlike OOV)"


def test_cold_oov_cache(make_model):
    model, _, table = make_model(use_item_id=True, id_dropout=0.0)
    cold_mask = torch.zeros(table.num_items, dtype=torch.bool)
    cold_mask[5] = True
    model.refresh_item_cache()
    warm5 = model.item_cache[5].clone()
    warm6 = model.item_cache[6].clone()
    model.refresh_item_cache(cold_mask=cold_mask)
    model.eval()
    with torch.no_grad():
        oov5 = model.item_tower(model._gather(torch.tensor([5])),
                                torch.tensor([model.oov_idx]))[0]
    assert torch.allclose(model.item_cache[5], oov5), "a cold item must be encoded with id -> OOV"
    assert not torch.allclose(model.item_cache[5], warm5), "the cold cache must differ from warm (real id)"
    assert torch.allclose(model.item_cache[6], warm6), "warm items must not change"


def test_history_source_grad(make_model):
    # cache: no gradient reaches the item tower through the history path
    model, _, _ = make_model(history_source="cache")
    model.refresh_item_cache()
    U = model.encode_users(_user_batch())
    U.sum().backward()
    assert all(p.grad is None for p in model.item_tower.parameters()), \
        "cache mode: the item tower must get no gradient from the user side"

    # embed: gradients flow into the hist_emb table
    model2, _, _ = make_model(history_source="embed")
    U2 = model2.encode_users(_user_batch())
    U2.sum().backward()
    g = model2.hist_emb.weight.grad
    assert g is not None and g.abs().sum() > 0, "embed mode: hist_emb must get gradients"
    assert g[0].abs().sum() == 0, "padding_idx=0 must get no gradient"


# --- synopsis (item-side text embedding) ---

def test_synopsis_off_by_default(make_model):
    model, _, _ = make_model()
    assert not model.item_tower.use_synopsis, "use_synopsis must default to False"
    assert not hasattr(model.item_tower, "synopsis_proj"), "synopsis off -> no projection"


def test_synopsis_builds_and_finite(make_model):
    model, _, _ = make_model(use_synopsis=True, synopsis_dim=12)
    assert model.item_tower.use_synopsis, "use_synopsis=True must turn the synopsis branch on"
    model.eval()
    with torch.no_grad():
        v = model.encode_items(torch.tensor([0, 1, 2, 3, 4]))   # includes PAD/OOV + low-info raw=0
    assert v.shape == (5, model.d)
    assert torch.isfinite(v).all(), "encode_items with synopsis must be finite (zero/low-info rows too)"


def test_synopsis_proj_affects_only_nonlowinfo(make_model):
    """Perturbing the projection changes the row with a real synopsis (idx 3); the low-info row
    (idx 4) uses no_synopsis and does not change. Mirrors the studios empty-vs-pad test."""
    model, _, _ = make_model(use_synopsis=True, synopsis_dim=12)
    model.eval()
    real, low = torch.tensor([3]), torch.tensor([4])
    with torch.no_grad():
        before_real, before_low = model.encode_items(real).clone(), model.encode_items(low).clone()
        model.item_tower.synopsis_proj[-1].weight += 5.0
        after_real, after_low = model.encode_items(real), model.encode_items(low)
    assert not torch.allclose(before_real, after_real), "a real-synopsis row must change with the projection"
    assert torch.allclose(before_low, after_low), "a low-info row uses no_synopsis; the projection has no effect"


def test_no_synopsis_param_affects_only_lowinfo(make_model):
    """The converse: perturbing no_synopsis changes the low-info row (idx 4), not the real one (idx 3)."""
    model, _, _ = make_model(use_synopsis=True, synopsis_dim=12)
    model.eval()
    real, low = torch.tensor([3]), torch.tensor([4])
    with torch.no_grad():
        before_real, before_low = model.encode_items(real).clone(), model.encode_items(low).clone()
        model.item_tower.no_synopsis += 5.0
        after_real, after_low = model.encode_items(real), model.encode_items(low)
    assert not torch.allclose(before_low, after_low), "a low-info row must change with no_synopsis (learned)"
    assert torch.allclose(before_real, after_real), "a real-synopsis row does not use no_synopsis"
