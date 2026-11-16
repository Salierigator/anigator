"""The bundle encoder (inference.UserEncoder) must give the same user vectors as TwoTower from the
same weights, for every pooling variant and with an empty history in the batch."""
import pytest
import torch

from model.twotower.retriever.inference import UserEncoder


@pytest.mark.parametrize("history_source", ["cache", "embed"])
@pytest.mark.parametrize("score_pool", ["none", "linear", "learned"])
@pytest.mark.parametrize("history_pool", ["mean", "attn"])
def test_user_encoder_matches_two_tower(make_model, spec, history_pool, score_pool, history_source):
    model, cfg, _ = make_model(history_pool=history_pool, score_pool=score_pool,
                               history_source=history_source)
    model.refresh_item_cache()
    model.eval()
    batch = {
        "history_ids": torch.tensor([[2, 3, 5, 0], [4, 0, 0, 0], [0, 0, 0, 0]]),
        "history_mask": torch.tensor([[True, True, True, False], [True, False, False, False],
                                      [False, False, False, False]]),
        "history_scores": torch.tensor([[10, 7, 0, 0], [3, 0, 0, 0], [0, 0, 0, 0]]),
        "gender_id": torch.tensor([1, 2, 0]),
        "joined_bucket": torch.tensor([0, 3, 4]),
    }
    meta = {"d": cfg.d, "mlp_hidden": cfg.mlp_hidden, "history_pool": history_pool,
            "score_pool": score_pool, "history_source": history_source,
            "user_features": spec["user_features"]}
    enc = UserEncoder(meta, model.item_cache.numpy())
    enc.load_state_dict({k: v for k, v in model.state_dict().items()
                         if not k.startswith("item_tower.")}, strict=True)
    enc.eval()

    with torch.no_grad():
        expected = model.encode_users(batch)
    got = enc.encode(batch["history_ids"], batch["history_mask"], batch["history_scores"],
                     batch["gender_id"], batch["joined_bucket"])
    assert torch.isfinite(got).all()
    assert torch.equal(got, expected)
