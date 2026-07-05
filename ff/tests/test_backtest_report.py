import pandas as pd
import pytest

from ff_model.backtest_report import build_position_report


def _backtest_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_id": "p1", "target_season": 2022, "full_projection_p50": 4.0, "naive_full_projection": 3.0, "adp": 1.0, "actual_fantasy_points": 40.0},
            {"player_id": "p2", "target_season": 2022, "full_projection_p50": 3.0, "naive_full_projection": 4.0, "adp": 2.0, "actual_fantasy_points": 30.0},
            {"player_id": "p3", "target_season": 2022, "full_projection_p50": 2.0, "naive_full_projection": 1.0, "adp": 3.0, "actual_fantasy_points": 20.0},
            {"player_id": "p4", "target_season": 2022, "full_projection_p50": 1.0, "naive_full_projection": 2.0, "adp": 4.0, "actual_fantasy_points": 10.0},
            # No ADP -- a bench player irrelevant enough that the crowd never drafted them.
            {"player_id": "p5", "target_season": 2022, "full_projection_p50": 999.0, "naive_full_projection": 999.0, "adp": None, "actual_fantasy_points": 1.0},
        ]
    )


def _disagreement_frame(n: int = 20, seasons: tuple[int, ...] = (2022, 2023)) -> pd.DataFrame:
    """Every player's model rank is the full reversal of their ADP rank (adp and
    full_projection_p50 share the same ascending values, so `rank(ascending=False)`
    on the model column flips the order), and the actual outcome matches the model's
    rank exactly -- a maximal, unambiguous "the model's disagreements are all correct"
    scenario, repeated identically across two seasons to exercise the season-block CI."""
    rows = []
    for season in seasons:
        for i in range(1, n + 1):
            rows.append(
                {
                    "player_id": f"s{season}_p{i}",
                    "target_season": season,
                    "full_projection_p50": float(i),
                    "naive_full_projection": float(i),
                    "adp": float(i),
                    "actual_fantasy_points": float(i),
                }
            )
    return pd.DataFrame(rows)


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


def test_report_states_the_disagreement_edge_verdict() -> None:
    """Per ADR-0014/issue #19: the report must state the Disagreement Edge win-condition
    verdict per position -- computed on the Matched Population, with season-block
    bootstrap CIs since the season (not the player) is the real unit of independence."""
    report = build_position_report(_disagreement_frame())

    edge = report["disagreement_edge"]
    # The model fully reverses ADP's rank order on every player, and actual outcomes
    # match the model's rank exactly -> the model's disagreements are all correct.
    assert edge["correlation"] == pytest.approx(1.0)
    assert edge["correlation_ci"][0] == pytest.approx(1.0)
    assert edge["correlation_ci"][1] == pytest.approx(1.0)
    assert edge["win_rate"] == pytest.approx(1.0)
    assert edge["win_rate_ci"][0] == pytest.approx(1.0)
    assert edge["win_rate_ci"][1] == pytest.approx(1.0)
    # Ranks are computed within each `target_season` group (draft rank only means
    # something relative to players drafted the same season), so for each season
    # |model_rank - adp_rank| = |21 - 2i| > 12 for i in {1,2,3,4,17,18,19,20} -- 8
    # players/season x 2 seasons.
    assert edge["n_large_disagreements"] == 16


def test_report_states_the_tier_accuracy_verdict() -> None:
    """Per ADR-0011/issue #14: the report must state a tier-accuracy beat/tied/lost
    verdict per position, as a supporting signal alongside Disagreement Edge, on the
    same Matched Population."""
    report = build_position_report(_disagreement_frame())

    tiers = report["tier_accuracy"]
    # The model's rank matches the actual outcome exactly, so it places every
    # player in his true tier; ADP's fully-reversed rank does worse.
    assert tiers["model"]["accuracy"] == pytest.approx(1.0)
    assert tiers["model"]["off_by_one_rate"] == pytest.approx(0.0)
    assert tiers["adp"]["accuracy"] < 1.0
    assert tiers["verdict"] == "beat"
