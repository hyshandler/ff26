import pandas as pd

from ff_model.features import (
    add_trailing_player_averages,
    add_trailing_team_shares,
    season_ending_averages,
    season_ending_shares,
)
from ff_model.position_config import PositionConfig
from ff_model.quantile_model import predict_quantiles, train_quantile_models


def _average_stat_columns(config: PositionConfig) -> dict[str, str]:
    return {"trailing_snap_pct": "offense_pct"} | {
        f"trailing_avg_{stat}": stat for stat in config.raw_stat_columns
    }


def feature_columns(config: PositionConfig) -> list[str]:
    """Process/opportunity features first (primary split candidates for LightGBM),
    then trailing raw-stat averages as secondary, lower-priority features."""
    return (
        list(config.share_stat_columns)
        + ["trailing_snap_pct"]
        + [f"trailing_avg_{stat}" for stat in config.raw_stat_columns]
    )


def _with_red_zone_carries(
    config: PositionConfig, weekly_all_positions: pd.DataFrame, red_zone_carries: pd.DataFrame
) -> pd.DataFrame:
    if not config.needs_red_zone_data:
        return weekly_all_positions
    weekly = weekly_all_positions.merge(
        red_zone_carries, on=["season", "week", "player_id"], how="left"
    )
    weekly["red_zone_carries"] = weekly["red_zone_carries"].fillna(0).astype(float)
    return weekly


def add_position_features(
    config: PositionConfig,
    weekly_all_positions: pd.DataFrame,
    red_zone_carries: pd.DataFrame,
    snap_pct: pd.DataFrame,
) -> pd.DataFrame:
    """Player-weeks for `config.position`, with leakage-safe trailing features.

    Team totals for share features are computed from every position (a QB
    scramble or WR jet sweep still counts against the team's carries), so this
    must run before filtering down to this position's rows.
    """
    weekly = _with_red_zone_carries(config, weekly_all_positions, red_zone_carries)
    weekly = add_trailing_team_shares(weekly, config.share_stat_columns)

    weekly = weekly.merge(snap_pct, on=["season", "week", "player_id"], how="left")
    weekly = add_trailing_player_averages(weekly, _average_stat_columns(config))

    return weekly.loc[weekly["position"] == config.position]


def _prediction_features(
    config: PositionConfig,
    weekly_all_positions: pd.DataFrame,
    red_zone_carries: pd.DataFrame,
    snap_pct: pd.DataFrame,
    season: int,
    player_ids: set[str],
) -> pd.DataFrame:
    """Features a Veteran carries into `season + 1`, from `season`'s completed totals."""
    weekly = _with_red_zone_carries(config, weekly_all_positions, red_zone_carries)
    weekly = weekly.merge(snap_pct, on=["season", "week", "player_id"], how="left")

    shares = season_ending_shares(weekly, config.share_stat_columns, season=season)
    averages = season_ending_averages(weekly, _average_stat_columns(config), season=season)

    features = shares.merge(averages, on="player_id", how="left")
    return features.loc[features["player_id"].isin(player_ids)]


def build_position_model_projections(
    config: PositionConfig,
    weekly_all_positions: pd.DataFrame,
    red_zone_carries: pd.DataFrame,
    snap_pct: pd.DataFrame,
    train_through_season: int,
    target_season: int,
    eligible_player_ids: set[str],
) -> pd.DataFrame:
    """Train a per-quantile LightGBM model per raw stat, and project each eligible Veteran."""
    training = add_position_features(config, weekly_all_positions, red_zone_carries, snap_pct)
    training = training.loc[training["season"] <= train_through_season]

    prediction_features = _prediction_features(
        config,
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

    columns = feature_columns(config)
    X_train = training[columns]
    X_predict = prediction_features[columns]
    for stat in config.raw_stat_columns:
        models = train_quantile_models(X_train, training[stat])
        quantiles = predict_quantiles(models, X_predict)
        quantiles.index = prediction_features.index
        for column in quantiles.columns:
            output[f"{stat}_{column}"] = quantiles[column]

    return output
