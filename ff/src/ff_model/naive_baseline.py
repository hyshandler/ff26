import pandas as pd

from ff_model.data_availability import assert_no_participation_sourced_features

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
    weekly: pd.DataFrame, player_ids: set[str], through_season: int
) -> pd.DataFrame:
    """Predict per-game raw stats as each player's trailing average through `through_season`.

    This is the naive statistical baseline the real LightGBM model replaces in a
    later slice: it exists to prove the ingestion -> feature -> model -> backtest
    -> output pipeline shape end-to-end before real modeling effort goes in.
    """
    history = weekly.loc[
        (weekly["season"] <= through_season) & (weekly["player_id"].isin(player_ids))
    ]

    grouped = history.groupby("player_id")
    predictions = grouped[RAW_STAT_COLUMNS].mean()
    predictions["games"] = grouped.size()

    return predictions.reset_index()
