import pandas as pd

from ff_model.adp import crosswalk_adp_to_player_ids, load_adp
from ff_model.backtest import walk_forward_splits
from ff_model.backtest_report import build_position_report
from ff_model.experience_features import ExperienceFeature
from ff_model.features import MultiSeasonWindow
from ff_model.naive_baseline import RAW_STAT_COLUMNS, predict_naive_baseline, score_naive_baseline
from ff_model.nflverse import SNAP_COUNTS_EARLIEST_SEASON, load_seasonal_rosters, load_weekly_stats
from ff_model.pipeline import build_position_projections
from ff_model.position_model import ModelBackend
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

ADP_LATEST_AVAILABLE_SEASON = 2024
"""Last target season Fantasy Football Calculator has ADP for (per ADR-0007 /
`docs/research/historical-adp-and-projections-availability.md`). 2025 has no FFC ADP
at all -- confirmed permanent gap, tracked separately by issue #18's FantasyPros
scrape fallback -- so `build_backtest_report` never requests ADP past this season and
those rows simply fall outside the Matched Population rather than erroring."""


def run_backtest(
    position: str,
    seasons: list[int],
    min_train_seasons: int,
    include_depth_chart_competition: bool = False,
    multi_season_window: MultiSeasonWindow | None = None,
    experience_feature: ExperienceFeature | None = None,
    sos_feature: SosFeature | None = None,
    model_backend: ModelBackend = "lightgbm",
    permute_columns: list[str] | None = None,
    permute_random_state: int = 0,
) -> pd.DataFrame:
    """Run the Walk-Forward Backtest for one position and concatenate every split's output.

    `permute_columns`, if given, is passed through to `build_position_projections` on every
    split -- permutation feature importance (issue #26) applied consistently across the
    whole backtest range, not just one split.
    """
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
            model_backend=model_backend,
            permute_columns=permute_columns,
            permute_random_state=permute_random_state,
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


def build_backtest_report(
    position: str,
    seasons: list[int] = STANDARD_BACKTEST_SEASONS,
    min_train_seasons: int = 3,
    raw_stat_columns: list[str] = RAW_STAT_COLUMNS,
    formula: ScoringFormula = PPR,
    include_depth_chart_competition: bool = False,
    multi_season_window: MultiSeasonWindow | None = None,
    experience_feature: ExperienceFeature | None = None,
    sos_feature: SosFeature | None = None,
    model_backend: ModelBackend = "lightgbm",
    permute_columns: list[str] | None = None,
    permute_random_state: int = 0,
) -> dict:
    """Runs the Walk-Forward Backtest for `position` end-to-end and scores it per
    ADR-0010/issue #13: Answer Key actual outcomes, ADP Benchmark, and the naive
    baseline all joined on, then `build_position_report` leads the result with
    Matched Population rho (full-population rho demoted to context-only).

    ADP is only requested through `ADP_LATEST_AVAILABLE_SEASON` -- target seasons
    past it (e.g. 2025) get no ADP call at all and simply fall outside the Matched
    Population, rather than erroring on a season FFC has no data for.

    The feature-family knobs (`include_depth_chart_competition`, `multi_season_window`,
    `experience_feature`, `sos_feature`) and `model_backend` are forwarded straight to
    `run_backtest` -- left at their defaults this reproduces each position's adopted
    `PositionConfig`, but a caller can override them one at a time to score a feature-
    family variant on Disagreement Edge.
    """
    result = run_backtest(
        position,
        seasons,
        min_train_seasons,
        include_depth_chart_competition=include_depth_chart_competition,
        multi_season_window=multi_season_window,
        experience_feature=experience_feature,
        sos_feature=sos_feature,
        model_backend=model_backend,
        permute_columns=permute_columns,
        permute_random_state=permute_random_state,
    )

    # `with_naive_baseline` needs weekly rows through each split's `train_through_season`
    # (always earlier than that split's `target_season`), so load the full requested
    # `seasons` range rather than just the target seasons `with_actual_outcomes` needs.
    weekly = load_weekly_stats(sorted(set(seasons)))
    result = with_actual_outcomes(result, weekly, formula)
    result = with_naive_baseline(result, weekly, raw_stat_columns, formula)

    adp_target_seasons = sorted(s for s in set(result["target_season"]) if s <= ADP_LATEST_AVAILABLE_SEASON)
    if adp_target_seasons:
        rosters = load_seasonal_rosters(adp_target_seasons)
        for season in adp_target_seasons:
            adp = crosswalk_adp_to_player_ids(load_adp(season), rosters, season=season)
            result = with_adp_benchmark(result, adp, season=season)
    else:
        result["adp"] = pd.NA

    return build_position_report(result)
