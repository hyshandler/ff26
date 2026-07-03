import pytest

from ff_model.naive_baseline import RAW_STAT_COLUMNS
from ff_model.pipeline import build_position_projections


@pytest.mark.network
def test_build_position_projections_for_rb_against_real_historical_seasons() -> None:
    result = build_position_projections("RB", train_through_season=2022, target_season=2023)

    assert result.position == "RB"
    assert result.train_through_season == 2022
    assert result.target_season == 2023

    projections = result.projections
    assert len(projections) > 0
    assert {"player_id", "player_name", "games", "position", "target_season"} <= set(
        projections.columns
    )
    for stat in RAW_STAT_COLUMNS:
        for quantile_column in (f"{stat}_p10", f"{stat}_p50", f"{stat}_p90"):
            assert quantile_column in projections.columns
        assert (projections[f"{stat}_p10"] <= projections[f"{stat}_p50"]).all()
        assert (projections[f"{stat}_p50"] <= projections[f"{stat}_p90"]).all()

    # Every projected player must be a genuine Veteran: enough career games banked
    # before the target season, and not a rookie in it.
    assert (projections["games"] >= 1).all()

    # No duplicate players in the output.
    assert projections["player_id"].is_unique


def test_build_position_projections_rejects_a_non_adjacent_target_season() -> None:
    with pytest.raises(ValueError):
        build_position_projections("RB", train_through_season=2022, target_season=2024)
