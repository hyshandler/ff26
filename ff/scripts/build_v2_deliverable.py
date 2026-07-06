"""Issue #17: build the v2 deliverable -- 2026 projections for all four positions
(trained through the completed 2025 season), joined against live Fantasy Football
Calculator 2026 ADP, plus the v2 backtest report (Disagreement Edge per ADR-0014,
with matched-population Spearman and tier accuracy as supporting metrics only).

Writes:
- `output/v2_projections.csv` -- the deliverable itself.
- `output/backtest_report_v2.json` -- per-position report, consumed by
  `notebooks/build_notebook_v2.py`.
"""

import json
from pathlib import Path

import pandas as pd

from ff_model.adp import crosswalk_adp_to_player_ids, load_adp
from ff_model.disagreement_edge import LARGE_DISAGREEMENT_THRESHOLD, model_and_adp_ranks
from ff_model.nflverse import load_seasonal_rosters
from ff_model.pipeline import build_position_projections, combine_position_projections
from ff_model.position_config import POSITION_CONFIGS
from ff_model.run_backtest import STANDARD_BACKTEST_SEASONS, build_backtest_report

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"

TRAIN_THROUGH_SEASON = 2025
TARGET_SEASON = 2026

ROOKIE_EXCLUSION_NOTE = (
    "# Rookies are excluded per ADR-0006: this model only projects Veterans "
    "(players with at least one prior season and >=16 career games before the "
    "target season). See docs/adr/0006-rookies-excluded-from-v1.md.\n"
)


def _with_adp_ranks(projections: pd.DataFrame, adp: pd.DataFrame) -> pd.DataFrame:
    """Joins crosswalked live ADP onto `projections` and adds side-by-side ranks.

    Ranks (and the large-disagreement flag) are computed per position, over rows
    with non-null `adp` only -- ADP ranks a player relative to the field that was
    actually drafted, so pooling in undraftable depth players would be meaningless.
    Uses the same `LARGE_DISAGREEMENT_THRESHOLD` Disagreement Edge's win-rate uses
    (per issue #17), so a flagged disagreement here is the same thing the backtest
    report's win-rate is scored on, not a separate ad hoc definition.
    """
    result = projections.merge(
        adp[["player_id", "adp"]], on="player_id", how="left"
    )
    result["model_rank"] = pd.NA
    result["adp_rank"] = pd.NA
    result["rank_diff"] = pd.NA
    result["large_disagreement"] = False

    for position in result["position"].unique():
        mask = (result["position"] == position) & result["adp"].notna()
        if not mask.any():
            continue
        model_rank, adp_rank = model_and_adp_ranks(
            result.loc[mask], model_col="full_projection_p50", adp_col="adp"
        )
        rank_diff = model_rank - adp_rank
        result.loc[mask, "model_rank"] = model_rank
        result.loc[mask, "adp_rank"] = adp_rank
        result.loc[mask, "rank_diff"] = rank_diff
        result.loc[mask, "large_disagreement"] = rank_diff.abs() > LARGE_DISAGREEMENT_THRESHOLD

    return result


def build_projections_csv() -> pd.DataFrame:
    position_projections = []
    for position in POSITION_CONFIGS:
        position_projections.append(
            build_position_projections(position, TRAIN_THROUGH_SEASON, TARGET_SEASON)
        )
        print(f"  {position} projections done.", flush=True)
    combined = combine_position_projections(position_projections)

    live_adp = load_adp(TARGET_SEASON)
    rosters = load_seasonal_rosters([TARGET_SEASON])
    crosswalked_adp = crosswalk_adp_to_player_ids(live_adp, rosters, season=TARGET_SEASON)

    return _with_adp_ranks(combined, crosswalked_adp)


def build_backtest_reports() -> dict:
    reports = {}
    for position in POSITION_CONFIGS:
        reports[position] = build_backtest_report(position, seasons=STANDARD_BACKTEST_SEASONS)
        print(f"  {position} backtest report done.", flush=True)
    return reports


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Building 2026 projections...")
    projections = build_projections_csv()
    csv_path = OUTPUT_DIR / "v2_projections.csv"
    with open(csv_path, "w") as f:
        f.write(ROOKIE_EXCLUSION_NOTE)
        projections.to_csv(f, index=False)
    print(f"Wrote {len(projections)} rows to {csv_path}")

    print("Building v2 backtest report (2012-2025, per position)...")
    reports = build_backtest_reports()
    report_path = OUTPUT_DIR / "backtest_report_v2.json"
    report_path.write_text(json.dumps(reports, indent=2))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
