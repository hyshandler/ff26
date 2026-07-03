import pandas as pd

from ff_model.data_availability import assert_no_participation_sourced_features
from ff_model.scoring import ScoringFormula

RAW_STAT_COLUMNS = [
    "carries",
    "rushing_yards",
    "rushing_tds",
    "targets",
    "receptions",
    "receiving_yards",
    "receiving_tds",
]

assert_no_participation_sourced_features(RAW_STAT_COLUMNS)


def predict_naive_baseline(
    weekly: pd.DataFrame,
    player_ids: set[str],
    through_season: int,
    raw_stat_columns: list[str] = RAW_STAT_COLUMNS,
) -> pd.DataFrame:
    """Predict per-game raw stats as each player's trailing average through `through_season`.

    This is the naive statistical baseline the real LightGBM model replaces in a
    later slice: it exists to prove the ingestion -> feature -> model -> backtest
    -> output pipeline shape end-to-end before real modeling effort goes in.
    `raw_stat_columns` defaults to RB's stats, but any position's
    `PositionConfig.raw_stat_columns` can be passed to reuse this for QB/WR/TE too.
    """
    history = weekly.loc[
        (weekly["season"] <= through_season) & (weekly["player_id"].isin(player_ids))
    ]

    grouped = history.groupby("player_id")
    predictions = grouped[raw_stat_columns].mean()
    predictions["games"] = grouped.size()

    return predictions.reset_index()


def score_naive_baseline(predictions: pd.DataFrame, formula: ScoringFormula) -> pd.Series:
    """Fantasy points per game for the naive baseline's trailing-average raw stats.

    Unlike `scoring.score_projections` (which reads quantile-suffixed columns), the
    naive baseline is a single point estimate per stat, so this applies `formula`
    directly to `predictions`' raw stat columns.
    """
    points = pd.Series(0.0, index=predictions.index)
    for stat, points_per_unit in formula.points_per_unit.items():
        if stat in predictions.columns:
            points = points + predictions[stat] * points_per_unit
    return points
