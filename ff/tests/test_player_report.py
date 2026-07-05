import pandas as pd
import pytest

from ff_model.player_report import player_report, top_players_report, with_predicted_adp_rank


def _combined_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"target_season": 2023, "position": "RB", "train_through_season": 2022, "player_name": "Test Player", "fantasy_points_p50": 10.0, "full_projection_p50": 12.0, "actual_fantasy_points": 15.0, "adp": 5.0},
            {"target_season": 2023, "position": "RB", "train_through_season": 2022, "player_name": "Other Guy", "fantasy_points_p50": 8.0, "full_projection_p50": 9.0, "actual_fantasy_points": 7.0, "adp": 2.0},
            {"target_season": 2024, "position": "RB", "train_through_season": 2023, "player_name": "Test Player", "fantasy_points_p50": 11.0, "full_projection_p50": 13.0, "actual_fantasy_points": 10.0, "adp": None},
        ]
    )


def test_with_predicted_adp_rank_ranks_within_season_across_positions() -> None:
    result = with_predicted_adp_rank(_combined_frame())
    season_2023 = result[result["target_season"] == 2023]
    # Test Player's 12.0 projection outranks Other Guy's 9.0 -> rank 1 vs 2.
    assert season_2023.set_index("player_name")["predicted_adp_rank"]["Test Player"] == 1.0
    assert season_2023.set_index("player_name")["predicted_adp_rank"]["Other Guy"] == 2.0


def test_player_report_covers_every_season_and_computes_errors() -> None:
    combined = with_predicted_adp_rank(_combined_frame())
    report = player_report(combined, "Test Player")

    assert list(report["target_season"]) == [2023, 2024]
    assert report.loc[0, "points_error"] == pytest.approx(12.0 - 15.0)
    assert report.loc[0, "adp_rank_error"] == pytest.approx(1.0 - 5.0)
    # No real ADP in 2024 -> adp_rank_error is NaN, not raised or dropped.
    assert pd.isna(report.loc[1, "adp"])
    assert pd.isna(report.loc[1, "adp_rank_error"])


def test_player_report_is_case_insensitive() -> None:
    combined = with_predicted_adp_rank(_combined_frame())
    report = player_report(combined, "test player")
    assert len(report) == 2


def test_player_report_raises_with_suggestion_for_unknown_player() -> None:
    combined = with_predicted_adp_rank(_combined_frame())
    with pytest.raises(ValueError, match="No backtest rows found"):
        player_report(combined, "Nobody Real")


def test_top_players_report_ranks_by_actual_points_desc() -> None:
    combined = with_predicted_adp_rank(_combined_frame())
    report = top_players_report(combined, 2023)
    assert list(report["player_name"]) == ["Test Player", "Other Guy"]
    assert report.loc[0, "points_error"] == pytest.approx(12.0 - 15.0)
    assert report.loc[0, "adp_rank_error"] == pytest.approx(1.0 - 5.0)


def test_top_players_report_respects_limit_and_position() -> None:
    combined = with_predicted_adp_rank(_combined_frame())
    report = top_players_report(combined, 2023, position="rb", limit=1)
    assert list(report["player_name"]) == ["Test Player"]


def test_top_players_report_raises_for_unknown_season() -> None:
    combined = with_predicted_adp_rank(_combined_frame())
    with pytest.raises(ValueError, match="No backtest rows found"):
        top_players_report(combined, 1999)
