"""RealService on a stub recommender: the service needs nothing beyond model.base."""
import pytest
from fastapi import HTTPException

import app.config  # noqa: F401  (puts the repo root on sys.path)
import model
from app.config import get_settings
from app.schemas.recommend import RecommendRequest
from app.services.real_service import RealService
from model.base import Recommendations, Recommender


class Stub(Recommender):
    def recommend(self, user, *, top_k=20, cold_k=10, exclude=(), sfw=True, anchor=None):
        if anchor == 0:
            raise KeyError(anchor)
        self.call = {"user": user, "exclude": list(exclude), "anchor": anchor}
        rows = [{"mal_id": i, "title": f"anime {i}", "type": "TV"} for i in range(1, top_k + 1)]
        return Recommendations(sections={"main": rows},
                               meta={"source": "mal_ids", "history_count": 2, "not_in_api": 1})

    def knows_anime(self, mal_id):
        return mal_id == 1


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(model, "load", lambda name: Stub())
    return RealService(get_settings())


def test_mal_ids_reach_the_recommender(service):
    out = service.recommend(RecommendRequest(mal_ids=[1, 2], top_k=3, exclude_ids=[9],
                                             anchor_mal_id=5))
    assert [r.mal_id for r in out.main] == [1, 2, 3]
    assert out.cold == []                                   # this stub has no cold section
    assert service.rec.call["user"].mal_ids == [1, 2]
    assert service.rec.call["exclude"] == [9] and service.rec.call["anchor"] == 5
    assert (out.meta.source, out.meta.history_count, out.meta.total_entries) == ("mal_ids", 2, 2)
    assert out.meta.mode == "live" and out.meta.alpha is None


def test_empty_request_is_a_guest(service):
    out = service.recommend(RecommendRequest())
    assert service.rec.call["user"].mal_ids == []
    assert out.meta.source == "guest"


def test_unknown_anchor_is_422(service):
    with pytest.raises(HTTPException) as e:
        service.recommend(RecommendRequest(mal_ids=[1], anchor_mal_id=0))
    assert e.value.status_code == 422
