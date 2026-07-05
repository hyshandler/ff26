"""Per-player Walk-Forward Backtest report: one player, every season the
precomputed backtest covers, the model's Projection vs. actual outcome, and
the model's rank-derived "predicted ADP" vs. the actual ADP Benchmark.

Reads already-computed `output/backtest_{POSITION}.csv` files (produced by
`ff_model.__main__` / `run_backtest`) rather than re-running the model, since
they already contain every Walk-Forward Backtest split's Projection, actual
outcome, and ADP Benchmark for every position.

"Predicted ADP" isn't something the model outputs directly (see CONTEXT.md's
ADP Blend entry -- ADP as a model input/output is out of scope for v1/v2).
It's derived here from the model's rank by `full_projection_p50`, pooled
across positions within each target season, as a stand-in for "where the
model would have drafted this player" to compare against the real ADP
Benchmark's rank. The pool is limited to the positions the model covers (per
ADR-0006/ADR-0010), so it under-covers the real draft field (which also
includes Rookies) -- read it as directionally comparable, not an exact ADP
prediction.
"""

from pathlib import Path

import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE"]

REPORT_COLUMNS = [
    "target_season",
    "position",
    "train_through_season",
    "fantasy_points_p50",
    "full_projection_p50",
    "actual_fantasy_points",
    "predicted_adp_rank",
    "adp",
]

TOP_PLAYERS_COLUMNS = [
    "target_season",
    "player_name",
    "position",
    "full_projection_p50",
    "actual_fantasy_points",
    "predicted_adp_rank",
    "adp",
]


def load_combined_backtest(output_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(output_dir / f"backtest_{position}.csv") for position in POSITIONS]
    return pd.concat(frames, ignore_index=True)


def with_predicted_adp_rank(combined: pd.DataFrame) -> pd.DataFrame:
    """Adds `predicted_adp_rank`: the row's dense rank by `full_projection_p50`
    (1 = highest projection), computed within its own `target_season` pooled
    across positions, so it's comparable to the real (also cross-position) ADP.
    """
    result = combined.copy()
    result["predicted_adp_rank"] = result.groupby("target_season")["full_projection_p50"].rank(
        ascending=False, method="dense"
    )
    return result


def player_report(combined: pd.DataFrame, player_name: str) -> pd.DataFrame:
    """Builds the per-player report: one row per season the backtest covers
    for that player, with Projection/actual and predicted-ADP-rank/actual-ADP
    errors. Player lookup is case-insensitive; if there's no exact match, falls
    back to a substring match and raises with close-name suggestions if that's
    empty too.
    """
    matches = combined.loc[combined["player_name"].str.lower() == player_name.lower()]
    if matches.empty:
        matches = combined.loc[
            combined["player_name"].str.lower().str.contains(player_name.lower(), na=False)
        ]
    if matches.empty:
        close = combined["player_name"].unique()
        suggestion = f" Did you mean one of: {', '.join(sorted(close)[:5])}?" if len(close) else ""
        raise ValueError(f"No backtest rows found for player_name={player_name!r}.{suggestion}")

    report = matches.sort_values("target_season")[REPORT_COLUMNS].copy()
    report["points_error"] = report["full_projection_p50"] - report["actual_fantasy_points"]
    report["adp_rank_error"] = report["predicted_adp_rank"] - report["adp"]
    return report.reset_index(drop=True)


def top_players_report(
    combined: pd.DataFrame,
    season: int,
    position: str | None = None,
    limit: int = 50,
) -> pd.DataFrame:
    """Builds the top-players-of-the-season report: the `limit` players with
    the highest `actual_fantasy_points` for `season` (optionally filtered to a
    single `position`), each with its Projection-vs-actual and predicted-ADP-
    rank-vs-actual-ADP errors.
    """
    rows = combined.loc[combined["target_season"] == season]
    if position is not None:
        rows = rows.loc[rows["position"].str.upper() == position.upper()]
    if rows.empty:
        raise ValueError(f"No backtest rows found for target_season={season!r}, position={position!r}.")

    report = (
        rows.sort_values("actual_fantasy_points", ascending=False)
        .head(limit)[TOP_PLAYERS_COLUMNS]
        .copy()
    )
    report["points_error"] = report["full_projection_p50"] - report["actual_fantasy_points"]
    report["adp_rank_error"] = report["predicted_adp_rank"] - report["adp"]
    return report.reset_index(drop=True)
