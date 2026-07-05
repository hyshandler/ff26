import pandas as pd
import pytest

from ff_model.backtest_report import build_position_report


def _backtest_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_id": "p1", "full_projection_p50": 4.0, "naive_full_projection": 3.0, "adp": 1.0, "actual_fantasy_points": 40.0},
            {"player_id": "p2", "full_projection_p50": 3.0, "naive_full_projection": 4.0, "adp": 2.0, "actual_fantasy_points": 30.0},
            {"player_id": "p3", "full_projection_p50": 2.0, "naive_full_projection": 1.0, "adp": 3.0, "actual_fantasy_points": 20.0},
            {"player_id": "p4", "full_projection_p50": 1.0, "naive_full_projection": 2.0, "adp": 4.0, "actual_fantasy_points": 10.0},
            # No ADP -- a bench player irrelevant enough that the crowd never drafted them.
            {"player_id": "p5", "full_projection_p50": 999.0, "naive_full_projection": 999.0, "adp": None, "actual_fantasy_points": 1.0},
        ]
    )


def test_report_leads_with_matched_population_and_labels_full_population_context_only() -> None:
    report = build_position_report(_backtest_frame())

    assert report["n"] == 5
    assert report["n_matched"] == 4
    assert set(report["matched_population"]) == {"model", "adp", "naive"}
    assert set(report["full_population_context_only"]) == {"model", "adp", "naive"}
    # ADP's orientation is flipped internally (lower ADP = better) -- perfectly co-ranked
    # with the Answer Key on the matched rows.
    assert report["matched_population"]["adp"] == pytest.approx(1.0)
    # Naive baseline is still reported alongside, not dropped.
    assert "naive_mae" in report
    assert "model_mae" in report
    # MAE is scored on the Matched Population too -- p5's wild 999-vs-1 miss (an
    # undraftable bench player) would otherwise swamp the number the same way
    # ADR-0010 says it would swamp full-population rho.
    assert report["model_mae"] == pytest.approx(22.5)
    assert report["naive_mae"] == pytest.approx(22.5)
