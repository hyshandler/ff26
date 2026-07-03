import pandas as pd

from ff_model.scoring import PPR, score_projections


def test_ppr_scores_a_known_stat_line_correctly() -> None:
    # 100 rushing yards (10 pts @ 0.1/yd), 1 rushing TD (6 pts), 5 receptions (5 pts @ 1/rec),
    # 50 receiving yards (5 pts @ 0.1/yd), 0 receiving TDs -> 10 + 6 + 5 + 5 = 26.
    projections = pd.DataFrame(
        [
            {
                "rushing_yards_p50": 100.0,
                "rushing_tds_p50": 1.0,
                "receptions_p50": 5.0,
                "receiving_yards_p50": 50.0,
                "receiving_tds_p50": 0.0,
            }
        ]
    )

    result = score_projections(projections, PPR)

    assert result["fantasy_points_p50"].iloc[0] == 26.0


def test_ppr_ignores_scored_stats_absent_from_the_position_projections() -> None:
    # An RB's projections have no passing_yards/passing_tds/interceptions columns at
    # all (QB-only stats) -- those weights must simply not contribute, not error.
    projections = pd.DataFrame([{"rushing_yards_p50": 100.0, "rushing_tds_p50": 0.0}])

    result = score_projections(projections, PPR)

    assert result["fantasy_points_p50"].iloc[0] == 10.0
