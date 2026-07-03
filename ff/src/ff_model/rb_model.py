import pandas as pd

from ff_model.data_availability import assert_no_participation_sourced_features
from ff_model.features import (
    add_trailing_player_averages,
    add_trailing_team_shares,
    season_ending_averages,
    season_ending_shares,
)
from ff_model.naive_baseline import RAW_STAT_COLUMNS
from ff_model.quantile_model import predict_quantiles, train_quantile_models

SHARE_STAT_COLUMNS = {"carry_share": "carries", "trailing_red_zone_share": "red_zone_carries"}
AVERAGE_STAT_COLUMNS = {"trailing_snap_pct": "offense_pct"} | {
    f"trailing_avg_{stat}": stat for stat in RAW_STAT_COLUMNS
}

# Process/opportunity signals first, so they're the primary split candidates for
# LightGBM; trailing raw-stat averages are included per the spec but ordered
# after them as secondary, lower-priority features.
FEATURE_COLUMNS = list(SHARE_STAT_COLUMNS) + ["trailing_snap_pct"] + [
    f"trailing_avg_{stat}" for stat in RAW_STAT_COLUMNS
]

assert_no_participation_sourced_features(RAW_STAT_COLUMNS)
assert_no_participation_sourced_features(FEATURE_COLUMNS)


def _with_red_zone_carries(
    weekly_all_positions: pd.DataFrame, red_zone_carries: pd.DataFrame
) -> pd.DataFrame:
    weekly = weekly_all_positions.merge(
        red_zone_carries, on=["season", "week", "player_id"], how="left"
    )
    weekly["red_zone_carries"] = weekly["red_zone_carries"].fillna(0).astype(float)
    return weekly


def add_rb_features(
    weekly_all_positions: pd.DataFrame,
    red_zone_carries: pd.DataFrame,
    snap_pct: pd.DataFrame,
) -> pd.DataFrame:
    """RB player-weeks with leakage-safe trailing process/opportunity features.

    Team totals for carry_share and trailing_red_zone_share are computed from
    every position (a QB scramble or WR jet sweep still counts against the
    team's carries), so this must run before filtering down to RB rows.
    """
    weekly = _with_red_zone_carries(weekly_all_positions, red_zone_carries)
    weekly = add_trailing_team_shares(weekly, SHARE_STAT_COLUMNS)

    weekly = weekly.merge(snap_pct, on=["season", "week", "player_id"], how="left")
    weekly = add_trailing_player_averages(weekly, AVERAGE_STAT_COLUMNS)

    return weekly.loc[weekly["position"] == "RB"]


def _prediction_features(
    weekly_all_positions: pd.DataFrame,
    red_zone_carries: pd.DataFrame,
    snap_pct: pd.DataFrame,
    season: int,
    player_ids: set[str],
) -> pd.DataFrame:
    """Features a Veteran carries into `season + 1`, from `season`'s completed totals."""
    weekly = _with_red_zone_carries(weekly_all_positions, red_zone_carries)
    weekly = weekly.merge(snap_pct, on=["season", "week", "player_id"], how="left")

    shares = season_ending_shares(weekly, SHARE_STAT_COLUMNS, season=season)
    averages = season_ending_averages(weekly, AVERAGE_STAT_COLUMNS, season=season)

    features = shares.merge(averages, on="player_id", how="left")
    return features.loc[features["player_id"].isin(player_ids)]


def build_rb_projections(
    weekly_all_positions: pd.DataFrame,
    red_zone_carries: pd.DataFrame,
    snap_pct: pd.DataFrame,
    train_through_season: int,
    target_season: int,
    eligible_player_ids: set[str],
) -> pd.DataFrame:
    """Train a per-quantile LightGBM model per raw stat, and project each eligible Veteran."""
    training = add_rb_features(weekly_all_positions, red_zone_carries, snap_pct)
    training = training.loc[training["season"] <= train_through_season]

    prediction_features = _prediction_features(
        weekly_all_positions,
        red_zone_carries,
        snap_pct,
        season=train_through_season,
        player_ids=eligible_player_ids,
    )

    output = prediction_features[["player_id"]].copy()
    games = (
        training.loc[training["player_id"].isin(eligible_player_ids)]
        .groupby("player_id")
        .size()
        .rename("games")
    )
    output["games"] = output["player_id"].map(games)

    X_train = training[FEATURE_COLUMNS]
    X_predict = prediction_features[FEATURE_COLUMNS]
    for stat in RAW_STAT_COLUMNS:
        models = train_quantile_models(X_train, training[stat])
        quantiles = predict_quantiles(models, X_predict)
        quantiles.index = prediction_features.index
        for column in quantiles.columns:
            output[f"{stat}_{column}"] = quantiles[column]

    return output
