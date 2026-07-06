"""Issue #26: rebuild the WR backtest report with all four round-1 feature families
(Prior-Season Total #22, Opportunity Vacuum #23, Own-Team Offensive Environment #24,
Per-Touch Efficiency #25) active, over the standard 2012-2025 Walk-Forward Backtest,
plus a permutation-importance sweep of each family's columns against the resulting
Disagreement Edge (ADR-0014, primary), matched-population Spearman, and tier accuracy
(both supporting).

Per ADR-0015, these four families were wired into `PositionConfig` directly rather
than through a pre-adoption Disagreement Edge sweep -- this is the load-bearing check
of whether they actually earn their keep. Writes raw results incrementally to
`assets/wr_feature_family_backtest_26.json` (one line of progress per config to
stdout) so a killed/resumed run doesn't lose completed configs, mirroring
`scripts/feature_family_sweep.py`'s pattern.
"""

import json
from pathlib import Path

from ff_model.per_touch_efficiency import PER_TOUCH_EFFICIENCY_COLUMNS
from ff_model.run_backtest import STANDARD_BACKTEST_SEASONS, build_backtest_report

POSITION = "WR"

FAMILIES = {
    "baseline": [],
    "prior_season_totals": ["prior_season_fantasy_points", "prior_season_games_played"],
    "opportunity_vacuum": ["vacated_target_share", "prior_season_points_per_target"],
    "team_offensive_environment": [
        "team_pass_attempts_per_game",
        "team_points_per_game",
        "team_passing_ypa",
    ],
    "per_touch_efficiency": PER_TOUCH_EFFICIENCY_COLUMNS,
}

OUT_PATH = Path(__file__).resolve().parents[1] / "assets" / "wr_feature_family_backtest_26.json"


def _summarize(report: dict) -> dict:
    return {
        "n_matched": report["n_matched"],
        "disagreement_edge": report["disagreement_edge"],
        "matched_population": report["matched_population"],
        "tier_accuracy": report["tier_accuracy"],
    }


def main() -> None:
    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}

    for family, columns in FAMILIES.items():
        if family in results:
            continue
        report = build_backtest_report(
            POSITION,
            seasons=STANDARD_BACKTEST_SEASONS,
            permute_columns=columns or None,
        )
        results[family] = _summarize(report)
        OUT_PATH.write_text(json.dumps(results, indent=2))
        edge = results[family]["disagreement_edge"]
        print(f"{POSITION} {family:<28} corr={edge['correlation']:.3f} ci={edge['correlation_ci']}")


if __name__ == "__main__":
    main()
