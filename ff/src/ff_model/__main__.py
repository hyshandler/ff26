import argparse
from pathlib import Path

from ff_model.run_backtest import run_backtest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Walk-Forward Backtest and write a CSV.")
    parser.add_argument("--position", default="RB")
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--min-train-seasons", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("rb_projections_backtest.csv"))
    args = parser.parse_args()

    result = run_backtest(args.position, args.seasons, args.min_train_seasons)
    result.to_csv(args.out, index=False)
    print(f"Wrote {len(result)} rows to {args.out}")


if __name__ == "__main__":
    main()
