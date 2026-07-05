import pandas as pd
import pytest
from fastapi.testclient import TestClient

import ff_model.api as api_module


@pytest.fixture(autouse=True)
def _stub_combined_backtest(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        [
            {"target_season": 2023, "position": "RB", "train_through_season": 2022, "player_name": "Test Player", "fantasy_points_p50": 10.0, "full_projection_p50": 12.0, "actual_fantasy_points": 15.0, "predicted_adp_rank": 1.0, "adp": 5.0},
            {"target_season": 2024, "position": "RB", "train_through_season": 2023, "player_name": "Test Player", "fantasy_points_p50": 11.0, "full_projection_p50": 13.0, "actual_fantasy_points": 10.0, "predicted_adp_rank": 1.0, "adp": None},
        ]
    )
    api_module._combined_backtest.cache_clear()
    monkeypatch.setattr(api_module, "_combined_backtest", lambda: frame)


@pytest.fixture
def client() -> TestClient:
    return TestClient(api_module.app)


def test_list_players_filters_by_query(client: TestClient) -> None:
    resp = client.get("/api/players", params={"q": "test"})
    assert resp.status_code == 200
    assert resp.json() == ["Test Player"]


def test_get_player_report_returns_rows_with_null_adp_as_json_null(client: TestClient) -> None:
    resp = client.get("/api/players/Test Player")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    assert rows[1]["adp"] is None


def test_get_player_report_404s_for_unknown_player(client: TestClient) -> None:
    resp = client.get("/api/players/Nobody Real")
    assert resp.status_code == 404
    assert "No backtest rows found" in resp.json()["detail"]


def test_index_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_list_seasons_returns_sorted_unique_years(client: TestClient) -> None:
    resp = client.get("/api/seasons")
    assert resp.status_code == 200
    assert resp.json() == [2023, 2024]


def test_get_top_players_ranks_by_actual_points_desc(client: TestClient) -> None:
    resp = client.get("/api/seasons/2023/top")
    assert resp.status_code == 200
    rows = resp.json()
    assert [r["player_name"] for r in rows] == ["Test Player"]


def test_get_top_players_respects_limit_and_position(client: TestClient) -> None:
    resp = client.get("/api/seasons/2023/top", params={"position": "rb", "limit": 1})
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_top_players_404s_for_unknown_season(client: TestClient) -> None:
    resp = client.get("/api/seasons/1999/top")
    assert resp.status_code == 404
    assert "No backtest rows found" in resp.json()["detail"]
