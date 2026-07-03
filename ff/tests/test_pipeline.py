import pandas as pd
import pytest

from ff_model.pipeline import (
    PositionProjections,
    build_position_projections,
    combine_position_projections,
)
from ff_model.position_config import POSITION_CONFIGS


@pytest.mark.network
def test_build_position_projections_for_rb_against_real_historical_seasons() -> None:
    result = build_position_projections("RB", train_through_season=2022, target_season=2023)

    assert result.position == "RB"
    assert result.train_through_season == 2022
    assert result.target_season == 2023

    projections = result.projections
    assert len(projections) > 0
    assert {
        "player_id",
        "player_name",
        "games",
        "position",
        "target_season",
        "games_played_estimate",
        "fantasy_points_p10",
        "fantasy_points_p50",
        "fantasy_points_p90",
    } <= set(projections.columns)
    assert projections["games_played_estimate"].between(0, 17).all()
    assert (projections["fantasy_points_p10"] <= projections["fantasy_points_p50"]).all()
    assert (projections["fantasy_points_p50"] <= projections["fantasy_points_p90"]).all()
    for stat in POSITION_CONFIGS["RB"].raw_stat_columns:
        for quantile_column in (f"{stat}_p10", f"{stat}_p50", f"{stat}_p90"):
            assert quantile_column in projections.columns
        assert (projections[f"{stat}_p10"] <= projections[f"{stat}_p50"]).all()
        assert (projections[f"{stat}_p50"] <= projections[f"{stat}_p90"]).all()

    # Every projected player must be a genuine Veteran: enough career games banked
    # before the target season, and not a rookie in it.
    assert (projections["games"] >= 1).all()

    # No duplicate players in the output.
    assert projections["player_id"].is_unique


@pytest.mark.network
def test_build_position_projections_for_wr_against_real_historical_seasons() -> None:
    result = build_position_projections("WR", train_through_season=2022, target_season=2023)

    assert result.position == "WR"
    projections = result.projections
    assert len(projections) > 0
    for stat in POSITION_CONFIGS["WR"].raw_stat_columns:
        assert (projections[f"{stat}_p10"] <= projections[f"{stat}_p50"]).all()
        assert (projections[f"{stat}_p50"] <= projections[f"{stat}_p90"]).all()


def test_build_position_projections_rejects_a_non_adjacent_target_season() -> None:
    with pytest.raises(ValueError):
        build_position_projections("RB", train_through_season=2022, target_season=2024)


def test_build_position_projections_rejects_an_unknown_position() -> None:
    with pytest.raises(ValueError):
        build_position_projections("K", train_through_season=2022, target_season=2023)


def test_combine_position_projections_unions_columns_across_positions() -> None:
    rb = PositionProjections(
        position="RB",
        train_through_season=2022,
        target_season=2023,
        projections=pd.DataFrame([{"player_id": "rb1", "rushing_yards_p50": 50.0}]),
    )
    qb = PositionProjections(
        position="QB",
        train_through_season=2022,
        target_season=2023,
        projections=pd.DataFrame([{"player_id": "qb1", "passing_yards_p50": 250.0}]),
    )

    combined = combine_position_projections([rb, qb])

    assert len(combined) == 2
    assert {"player_id", "rushing_yards_p50", "passing_yards_p50"} <= set(combined.columns)
    rb_row = combined.set_index("player_id").loc["rb1"]
    assert pd.isna(rb_row["passing_yards_p50"])
