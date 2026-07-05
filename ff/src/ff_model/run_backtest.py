import pandas as pd

from ff_model.backtest import walk_forward_splits
from ff_model.experience_features import ExperienceFeature
from ff_model.features import MultiSeasonWindow
from ff_model.naive_baseline import predict_naive_baseline, score_naive_baseline
from ff_model.nflverse import SNAP_COUNTS_EARLIEST_SEASON
from ff_model.pipeline import build_position_projections
from ff_model.scoring import PPR, ScoringFormula
from ff_model.strength_of_schedule import SosFeature

STANDARD_BACKTEST_SEASONS = list(range(SNAP_COUNTS_EARLIEST_SEASON, 2026))
"""Default season range for the Walk-Forward Backtest.

Starts at `SNAP_COUNTS_EARLIEST_SEASON` (2012) -- nflverse's Pro-Football-Reference-sourced
snap counts are the last of the model's feature sources to become available (draft picks and
play-by-play go back much further, injury reports to 2009), so this is the earliest split
where every split's training years have a fully-featured `trailing_snap_pct`, not a silently
missing one. Ends at 2025, the most recent completed season -- covered by nflverse's
`stats_player` release (see `ff_model.nflverse`); bump this once a newer season is confirmed
available.
"""


def run_backtest(
    position: str,
    seasons: list[int],
    min_train_seasons: int,
    include_depth_chart_competition: bool = True,
    multi_season_window: MultiSeasonWindow | None = None,
    experience_feature: ExperienceFeature | None = None,
    sos_feature: SosFeature | None = None,
) -> pd.DataFrame:
    """Run the Walk-Forward Backtest for one position and concatenate every split's output."""
    splits = walk_forward_splits(seasons, min_train_seasons)

    split_frames = []
    for train_through_season, target_season in splits:
        result = build_position_projections(
            position,
            train_through_season,
            target_season,
            include_depth_chart_competition=include_depth_chart_competition,
            multi_season_window=multi_season_window,
            experience_feature=experience_feature,
            sos_feature=sos_feature,
        )
        frame = result.projections.copy()
        frame.insert(1, "train_through_season", train_through_season)
        split_frames.append(frame)

    return pd.concat(split_frames, ignore_index=True)


def with_actual_outcomes(
    backtest_result: pd.DataFrame, weekly: pd.DataFrame, formula: ScoringFormula = PPR
) -> pd.DataFrame:
    """Joins each row's actual full-season outcome onto a `run_backtest` result, for
    comparison against the full Projection.

    Per ADR-0008's survivorship-bias concern: a player_id/target_season pair with no
    rows in `weekly` (e.g. hurt and never returned) gets zero games and zero points
    rather than being dropped, so the backtest comparison isn't biased toward players
    who stayed healthy all season.
    """
    stat_columns = [stat for stat in formula.points_per_unit if stat in weekly.columns]
    season_totals = weekly.groupby(["player_id", "season"]).agg(
        actual_games_played=("week", "size"), **{stat: (stat, "sum") for stat in stat_columns}
    )

    points = pd.Series(0.0, index=season_totals.index)
    for stat, points_per_unit in formula.points_per_unit.items():
        if stat in season_totals.columns:
            points = points + season_totals[stat] * points_per_unit
    season_totals["actual_fantasy_points"] = points
    season_totals = season_totals[["actual_games_played", "actual_fantasy_points"]].reset_index()

    result = backtest_result.merge(
        season_totals,
        left_on=["player_id", "target_season"],
        right_on=["player_id", "season"],
        how="left",
    ).drop(columns="season")
    result["actual_games_played"] = result["actual_games_played"].fillna(0)
    result["actual_fantasy_points"] = result["actual_fantasy_points"].fillna(0.0)
    return result


def with_adp_benchmark(backtest_result: pd.DataFrame, adp: pd.DataFrame, season: int) -> pd.DataFrame:
    """Joins the ADP Benchmark (per ADR-0007) onto a `run_backtest` result's rows for
    `season`, for comparison against the model's full Projection and the naive baseline.

    `adp` is expected already crosswalked to `player_id` (see `crosswalk_adp_to_player_ids`).
    Only rows with `target_season == season` are matched, since one call covers one
    split's ADP data; a player missing from the crosswalk gets a NaN `adp` rather than
    being dropped. Safe to call repeatedly, chaining one season at a time (as a full
    backtest's ADP join does) -- an existing `adp` column from an earlier call is kept
    for every row outside `season`, not reset.
    """
    season_adp = adp.dropna(subset=["player_id"])[["player_id", "adp"]]

    result = backtest_result.copy()
    if "adp" not in result.columns:
        result["adp"] = pd.NA
    mask = result["target_season"] == season
    matched = result.loc[mask, ["player_id"]].merge(season_adp, on="player_id", how="left")["adp"]
    matched.index = result.loc[mask].index
    result.loc[mask, "adp"] = matched
    return result


def with_naive_baseline(
    backtest_result: pd.DataFrame, weekly: pd.DataFrame, raw_stat_columns: list[str], formula: ScoringFormula = PPR
) -> pd.DataFrame:
    """Joins the naive trailing-average baseline's full-season projection onto a
    `run_backtest` result, as `naive_full_projection` -- the same per-game-rate x
    Games-Played Estimate shape as the model's full Projection, so the two are
    directly comparable in the backtest report.

    Computed per split, using only `weekly` rows through that row's own
    `train_through_season`, respecting the same walk-forward boundary the model
    itself trains within (no peeking at the target season).
    """
    result = backtest_result.copy()
    result["naive_ppg"] = pd.NA
    result["naive_full_projection"] = pd.NA
    for train_through_season in result["train_through_season"].unique():
        mask = result["train_through_season"] == train_through_season
        player_ids = set(result.loc[mask, "player_id"])
        naive_predictions = predict_naive_baseline(weekly, player_ids, train_through_season, raw_stat_columns)
        naive_ppg = score_naive_baseline(naive_predictions, formula)
        naive_ppg.index = naive_predictions["player_id"]
        mapped = result.loc[mask, "player_id"].map(naive_ppg)
        result.loc[mask, "naive_ppg"] = mapped.to_numpy()
        result.loc[mask, "naive_full_projection"] = mapped.to_numpy() * result.loc[mask, "games_played_estimate"].to_numpy()
    return result
