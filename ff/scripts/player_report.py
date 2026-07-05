"""CLI wrapper around `ff_model.player_report`: prints one player's
Walk-Forward Backtest history (Projection vs. actual, predicted-ADP-rank vs.
actual ADP) to the terminal. See `ff_model.player_report` for the underlying
logic, which is also used by the `ff_model.api` FastAPI app.
"""

import argparse
from pathlib import Path

import pandas as pd

from ff_model.player_report import load_combined_backtest, player_report, with_predicted_adp_rank


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-player Walk-Forward Backtest report: Projection vs. actual, predicted-ADP-rank vs. actual ADP."
    )
    parser.add_argument("player_name", help="Exact player name as it appears in nflverse rosters, e.g. 'Christian McCaffrey'.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory containing backtest_{POSITION}.csv files.")
    args = parser.parse_args()

    combined = with_predicted_adp_rank(load_combined_backtest(args.output_dir))
    try:
        report = player_report(combined, args.player_name)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(report.to_string(index=False))

    n_with_adp = report["adp"].notna().sum()
    print(f"\nSeasons covered: {len(report)}")
    print(f"Points MAE (full_projection_p50 vs actual_fantasy_points): {report['points_error'].abs().mean():.2f}")
    if n_with_adp:
        print(f"ADP-rank MAE ({n_with_adp} seasons with real ADP): {report['adp_rank_error'].abs().mean():.2f}")
    else:
        print("No seasons had real ADP for this player (never drafted highly enough to be listed).")


if __name__ == "__main__":
    main()
