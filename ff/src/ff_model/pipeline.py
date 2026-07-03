from dataclasses import dataclass

import pandas as pd

from ff_model.naive_baseline import predict_naive_baseline
from ff_model.nflverse import load_seasonal_rosters, load_weekly_stats
from ff_model.veterans import veteran_player_ids

MIN_CAREER_GAMES = 16
"""Minimum career games before the rookie season to count as a Veteran.

One full season's worth of games, per ADR-0006 — tunable, not fixed by the PRD.
"""

EARLIEST_SEASON = 2006
"""Earliest season fed into training, across the whole pipeline.

Matches the modern nflverse era used in prior fantasy-projection modeling work
(see docs/research/fantasy-football-projection-modeling.md, 2006-2020 range).
The Walk-Forward Backtest trains on seasons EARLIEST_SEASON..train_through_season
(an expanding window), predicts train_through_season + 1, and slides forward.
"""


@dataclass(frozen=True)
class PositionProjections:
    position: str
    train_through_season: int
    target_season: int
    projections: pd.DataFrame


def build_position_projections(
    position: str, train_through_season: int, target_season: int
) -> PositionProjections:
    """Ingest -> filter -> predict -> output, for one position and one Walk-Forward split.

    The naive statistical baseline stands in for the real per-position model until
    a later slice replaces it with LightGBM; this seam's shape (and output schema)
    stays the same when that happens.
    """
    if target_season != train_through_season + 1:
        raise ValueError(
            "target_season must be train_through_season + 1 for the Walk-Forward Backtest"
        )

    weekly_seasons = list(range(EARLIEST_SEASON, train_through_season + 1))
    weekly = load_weekly_stats(weekly_seasons)
    weekly = weekly.loc[weekly["position"] == position]

    rosters = load_seasonal_rosters([target_season])
    rosters = rosters.loc[rosters["position"] == position]

    eligible = veteran_player_ids(
        rosters, weekly, season=target_season, min_career_games=MIN_CAREER_GAMES
    )

    predictions = predict_naive_baseline(
        weekly, player_ids=eligible, through_season=train_through_season
    )

    player_names = rosters.drop_duplicates("player_id").set_index("player_id")["player_name"]
    predictions.insert(1, "player_name", predictions["player_id"].map(player_names))
    predictions.insert(0, "target_season", target_season)
    predictions.insert(0, "position", position)

    return PositionProjections(
        position=position,
        train_through_season=train_through_season,
        target_season=target_season,
        projections=predictions,
    )
