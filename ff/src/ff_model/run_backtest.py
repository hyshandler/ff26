import pandas as pd

from ff_model.backtest import walk_forward_splits
from ff_model.pipeline import build_position_projections
from ff_model.scoring import PPR, ScoringFormula


def run_backtest(position: str, seasons: list[int], min_train_seasons: int) -> pd.DataFrame:
    """Run the Walk-Forward Backtest for one position and concatenate every split's output."""
    splits = walk_forward_splits(seasons, min_train_seasons)

    split_frames = []
    for train_through_season, target_season in splits:
        result = build_position_projections(position, train_through_season, target_season)
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
