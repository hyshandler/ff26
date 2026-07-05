from typing import Literal

import pandas as pd

from ff_model.experience_features import EXPERIENCE_FEATURE_COLUMN, ExperienceFeature
from ff_model.features import (
    add_trailing_player_averages,
    add_trailing_team_shares,
    season_ending_averages,
    season_ending_shares,
)
from ff_model.position_config import PositionConfig
from ff_model.strength_of_schedule import SOS_FEATURE_COLUMN, SosFeature, add_actual_game_sos
from ff_model.quantile_model import predict_quantiles, train_quantile_models
from ff_model.rf_model import predict_rf_quantiles, train_rf_quantile_model
from ff_model.tabfm_model import predict_tabfm_quantiles, train_tabfm_model

ModelBackend = Literal["lightgbm", "tabfm", "random_forest"]


def _average_stat_columns(config: PositionConfig, sos_feature: SosFeature = "none") -> dict[str, str]:
    columns = {"trailing_snap_pct": "offense_pct"} | {
        f"trailing_avg_{stat}": stat for stat in config.raw_stat_columns
    }
    if sos_feature == "actual_games":
        # The raw per-week "this week's opponent's trailing points allowed" column
        # (injected by `add_actual_game_sos`) gets trailing-averaged the same way as
        # any other raw stat, automatically aligning to games the player actually played.
        columns = columns | {"trailing_sos_faced": "trailing_points_allowed"}
    return columns


def multi_season_stat_columns(config: PositionConfig) -> dict[str, str]:
    """feature_name -> raw_stat_column for the multi-season memory features, named
    distinctly from `_average_stat_columns`'s single-season `trailing_avg_*` so both
    can coexist and be ablated independently in a backtest."""
    return {f"multi_season_avg_{stat}": stat for stat in config.raw_stat_columns}


def feature_columns(
    config: PositionConfig,
    include_depth_chart_competition: bool = True,
    include_multi_season: bool = False,
    experience_feature: ExperienceFeature = "none",
    sos_feature: SosFeature = "none",
) -> list[str]:
    """Process/opportunity features first (primary split candidates for LightGBM),
    then trailing raw-stat averages as secondary, lower-priority features.

    `include_depth_chart_competition=False` produces the without-feature baseline
    used to backtest the Depth-Chart Competition Feature's effect (ADR-0004).
    `include_multi_season=True` adds the multi-season memory features (whichever
    window `multi_season_history` was built with) for the same kind of ablation.
    `experience_feature` adds one age/experience column (age, years in league, career
    games, or a career-stage bucket) for the same kind of ablation.
    `sos_feature` adds one Strength-of-Schedule column (season-wide schedule average, or
    a trailing average aligned to actual games played) for the same kind of ablation.
    """
    depth_chart_column = ["depth_chart_competition"] if include_depth_chart_competition else []
    multi_season_columns = (
        list(multi_season_stat_columns(config).keys()) if include_multi_season else []
    )
    experience_columns = (
        [EXPERIENCE_FEATURE_COLUMN[experience_feature]] if experience_feature != "none" else []
    )
    sos_columns = [SOS_FEATURE_COLUMN[sos_feature]] if sos_feature != "none" else []
    return (
        list(config.share_stat_columns)
        + ["trailing_snap_pct"]
        + depth_chart_column
        + multi_season_columns
        + experience_columns
        + sos_columns
        + [f"trailing_avg_{stat}" for stat in config.raw_stat_columns]
    )


def _with_depth_chart_competition(weekly: pd.DataFrame, depth_chart_history: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.merge(depth_chart_history, on=["season", "player_id"], how="left")
    weekly["depth_chart_competition"] = weekly["depth_chart_competition"].fillna(0).astype(int)
    return weekly


def _with_multi_season_history(weekly: pd.DataFrame, multi_season_history: pd.DataFrame) -> pd.DataFrame:
    return weekly.merge(multi_season_history, on=["season", "player_id"], how="left")


def _with_experience_history(weekly: pd.DataFrame, experience_history: pd.DataFrame) -> pd.DataFrame:
    return weekly.merge(experience_history, on=["season", "player_id"], how="left")


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
    depth_chart_history: pd.DataFrame,
    multi_season_history: pd.DataFrame | None = None,
    experience_history: pd.DataFrame | None = None,
    sos_history: pd.DataFrame | None = None,
    league_wide_trailing_points_allowed: pd.DataFrame | None = None,
    sos_feature: SosFeature = "none",
) -> pd.DataFrame:
    """Player-weeks for `config.position`, with leakage-safe trailing features.

    Team totals for share features are computed from every position (a QB
    scramble or WR jet sweep still counts against the team's carries), so this
    must run before filtering down to this position's rows.
    """
    weekly = _with_red_zone_carries(config, weekly_all_positions, red_zone_carries)
    weekly = add_trailing_team_shares(weekly, config.share_stat_columns)

    weekly = weekly.merge(snap_pct, on=["season", "week", "player_id"], how="left")
    if sos_feature == "actual_games" and league_wide_trailing_points_allowed is not None:
        weekly = add_actual_game_sos(weekly, league_wide_trailing_points_allowed)
    weekly = add_trailing_player_averages(weekly, _average_stat_columns(config, sos_feature))
    weekly = _with_depth_chart_competition(weekly, depth_chart_history)
    if multi_season_history is not None:
        weekly = _with_multi_season_history(weekly, multi_season_history)
    if experience_history is not None:
        weekly = _with_experience_history(weekly, experience_history)
    if sos_history is not None:
        weekly = weekly.merge(sos_history, on=["season", "player_id"], how="left")

    return weekly.loc[weekly["position"] == config.position]


def _prediction_features(
    config: PositionConfig,
    weekly_all_positions: pd.DataFrame,
    red_zone_carries: pd.DataFrame,
    snap_pct: pd.DataFrame,
    depth_chart_history: pd.DataFrame,
    season: int,
    target_season: int,
    player_ids: set[str],
    multi_season_history: pd.DataFrame | None = None,
    experience_history: pd.DataFrame | None = None,
    sos_history: pd.DataFrame | None = None,
    league_wide_trailing_points_allowed: pd.DataFrame | None = None,
    sos_feature: SosFeature = "none",
) -> pd.DataFrame:
    """Features a Veteran carries into `target_season`, from `season`'s completed totals."""
    weekly = _with_red_zone_carries(config, weekly_all_positions, red_zone_carries)
    weekly = weekly.merge(snap_pct, on=["season", "week", "player_id"], how="left")
    if sos_feature == "actual_games" and league_wide_trailing_points_allowed is not None:
        weekly = add_actual_game_sos(weekly, league_wide_trailing_points_allowed)

    shares = season_ending_shares(weekly, config.share_stat_columns, season=season)
    averages = season_ending_averages(weekly, _average_stat_columns(config, sos_feature), season=season)

    features = shares.merge(averages, on="player_id", how="left")

    # Unlike shares/averages (drawn from `season`'s completed totals), the depth-chart
    # flag reflects the competition a player faces heading INTO `target_season`.
    target_flags = depth_chart_history.loc[
        depth_chart_history["season"] == target_season, ["player_id", "depth_chart_competition"]
    ]
    features = features.merge(target_flags, on="player_id", how="left")
    features["depth_chart_competition"] = features["depth_chart_competition"].fillna(0).astype(int)

    # Like the depth-chart flag, the multi-season history is keyed by the season the
    # feature applies TO (`target_season`), already computed from strictly-prior seasons.
    if multi_season_history is not None:
        target_multi_season = multi_season_history.loc[
            multi_season_history["season"] == target_season
        ].drop(columns="season")
        features = features.merge(target_multi_season, on="player_id", how="left")

    if experience_history is not None:
        target_experience = experience_history.loc[
            experience_history["season"] == target_season
        ].drop(columns="season")
        features = features.merge(target_experience, on="player_id", how="left")

    if sos_history is not None:
        target_sos = sos_history.loc[sos_history["season"] == target_season].drop(columns="season")
        features = features.merge(target_sos, on="player_id", how="left")

    return features.loc[features["player_id"].isin(player_ids)]


def build_position_model_projections(
    config: PositionConfig,
    weekly_all_positions: pd.DataFrame,
    red_zone_carries: pd.DataFrame,
    snap_pct: pd.DataFrame,
    depth_chart_history: pd.DataFrame,
    train_through_season: int,
    target_season: int,
    eligible_player_ids: set[str],
    model_backend: ModelBackend = "lightgbm",
    include_depth_chart_competition: bool = True,
    multi_season_history: pd.DataFrame | None = None,
    experience_history: pd.DataFrame | None = None,
    experience_feature: ExperienceFeature = "none",
    sos_history: pd.DataFrame | None = None,
    league_wide_trailing_points_allowed: pd.DataFrame | None = None,
    sos_feature: SosFeature = "none",
) -> pd.DataFrame:
    """Train a quantile model per raw stat, and project each eligible Veteran.

    `model_backend` selects LightGBM (per ADR-0002) or TabFM (per ADR-0009's
    apples-to-apples comparison); both produce the same p10/p50/p90 output shape.
    `include_depth_chart_competition=False` produces the without-feature baseline
    used to backtest the Depth-Chart Competition Feature's effect (ADR-0004).
    `multi_season_history` (from one of `features.multi_season_*_averages`) opts
    into the multi-season memory features for the same kind of ablation; omitted
    (None) keeps the single-season `trailing_avg_*` behavior unchanged.
    `experience_history` (from one of `experience_features.*_history`), paired with
    `experience_feature` naming which column it carries, opts into an age/experience
    feature the same way.
    `sos_feature="season_wide"` uses `sos_history` (from
    `strength_of_schedule.season_wide_sos_history` + `season_wide_sos_by_player`), an
    external table merged like the others. `sos_feature="actual_games"` instead uses
    `league_wide_trailing_points_allowed` (from `strength_of_schedule.weekly_points_allowed`
    + `trailing_points_allowed`), joined per-week onto each player's own opponent and then
    trailing-averaged like any other raw stat -- automatically aligned to games actually
    played, unlike the season-wide schedule average.
    """
    include_multi_season = multi_season_history is not None
    training = add_position_features(
        config,
        weekly_all_positions,
        red_zone_carries,
        snap_pct,
        depth_chart_history,
        multi_season_history=multi_season_history,
        experience_history=experience_history,
        sos_history=sos_history,
        league_wide_trailing_points_allowed=league_wide_trailing_points_allowed,
        sos_feature=sos_feature,
    )
    training = training.loc[training["season"] <= train_through_season]

    prediction_features = _prediction_features(
        config,
        weekly_all_positions,
        red_zone_carries,
        snap_pct,
        depth_chart_history,
        season=train_through_season,
        target_season=target_season,
        player_ids=eligible_player_ids,
        multi_season_history=multi_season_history,
        experience_history=experience_history,
        sos_history=sos_history,
        league_wide_trailing_points_allowed=league_wide_trailing_points_allowed,
        sos_feature=sos_feature,
    )

    output = prediction_features[["player_id"]].copy()
    games = (
        training.loc[training["player_id"].isin(eligible_player_ids)]
        .groupby("player_id")
        .size()
        .rename("games")
    )
    output["games"] = output["player_id"].map(games)

    columns = feature_columns(
        config,
        include_depth_chart_competition,
        include_multi_season,
        experience_feature,
        sos_feature,
    )
    X_train = training[columns]
    X_predict = prediction_features[columns]
    for stat in config.raw_stat_columns:
        if model_backend == "lightgbm":
            models = train_quantile_models(X_train, training[stat])
            quantiles = predict_quantiles(models, X_predict)
        elif model_backend == "random_forest":
            rf_model = train_rf_quantile_model(X_train, training[stat])
            quantiles = predict_rf_quantiles(rf_model, X_predict)
        else:
            model = train_tabfm_model(X_train, training[stat])
            quantiles = predict_tabfm_quantiles(model, X_predict)
        quantiles.index = prediction_features.index
        for column in quantiles.columns:
            output[f"{stat}_{column}"] = quantiles[column]

    return output
