"""API contract in mock mode (MOCK_MODE=1): no model is loaded."""
from fastapi.testclient import TestClient

from app.main import app


def test_health_mock():
    with TestClient(app) as client:
        r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["mode"] == "mock"


def test_recommend_mock_shape_and_cut():
    with TestClient(app) as client:
        r = client.post("/api/recommend",
                        json={"username": "anyone", "top_k": 5, "cold_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["main"]) <= 5
    assert len(body["cold"]) <= 3
    assert body["meta"]["mode"] == "mock"
    assert {"popularity", "members", "start_date"} <= set(body["main"][0])  # client-side sort + card


def test_recommend_empty_body_is_guest():
    # neither username nor mal_ids -> guest (empty history), not a 422
    with TestClient(app) as client:
        r = client.post("/api/recommend", json={"top_k": 5})
    assert r.status_code == 200
    assert r.json()["main"]


def test_recommend_exclude_ids():
    with TestClient(app) as client:
        first = client.post("/api/recommend", json={"username": "anyone"}) \
            .json()["main"][0]["mal_id"]
        r = client.post("/api/recommend",
                        json={"username": "anyone", "exclude_ids": [first]})
    assert r.status_code == 200
    assert first not in [x["mal_id"] for x in r.json()["main"]]


def test_search_mock_substring():
    with TestClient(app) as client:
        r = client.get("/api/search", params={"q": "clannad"})
    assert r.status_code == 200
    results = r.json()["results"]
    assert results and results[0]["title"] == "Clannad"
    assert {"mal_id", "title", "in_corpus"} <= set(results[0])


def test_search_q_too_short():
    with TestClient(app) as client:
        r = client.get("/api/search", params={"q": "a"})
    assert r.status_code == 422


def test_anime_details_mapped(monkeypatch):
    import app.api.routes.anime as anime_route
    detail = {"mal_id": 52991, "title": "Sousou no Frieren", "score": 9.31,
              "type": "TV", "year": 2023, "genres": ["Adventure"], "studios": ["Madhouse"]}
    monkeypatch.setattr(anime_route, "fetch_details", lambda mid: detail)
    with TestClient(app) as client:
        r = client.get("/api/anime/52991")
    assert r.status_code == 200
    body = r.json()
    assert body["title"] == "Sousou no Frieren" and body["score"] == 9.31
    assert r.headers["cache-control"] == "public, max-age=86400"


def test_anime_details_unavailable(monkeypatch):
    import app.api.routes.anime as anime_route
    monkeypatch.setattr(anime_route, "fetch_details", lambda mid: None)
    with TestClient(app) as client:
        r = client.get("/api/anime/1")
    assert r.status_code == 404


def test_username_exists(monkeypatch):
    import app.clients.mal_api as mal_api
    monkeypatch.setattr(mal_api, "user_exists", lambda u: u == "realuser")
    with TestClient(app) as client:
        assert client.get("/api/users/realuser/exists").json() == {"exists": True}
        assert client.get("/api/users/ghostuser/exists").json() == {"exists": False}


def test_username_exists_indeterminate(monkeypatch):
    import app.clients.mal_api as mal_api
    monkeypatch.setattr(mal_api, "user_exists", lambda u: None)
    with TestClient(app) as client:
        r = client.get("/api/users/unknown-state-user/exists")
    assert r.status_code == 502
