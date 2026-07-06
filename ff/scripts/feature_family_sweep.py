"""Issue #16: re-sweep each adopted feature family per position, scored on Disagreement Edge
(ADR-0014) over the full 2012-2025 Walk-Forward Backtest, one family varied at a time against
the position's other current defaults (see `run_backtest.build_backtest_report`'s docstring).

Writes raw results incrementally to `assets/feature_sweep_16.json` (one line of progress per
config to stdout) so a killed/resumed run doesn't lose completed configs.
"""

import json
from pathlib import Path

from ff_model.position_config import POSITION_CONFIGS
from ff_model.run_backtest import STANDARD_BACKTEST_SEASONS, build_backtest_report

FAMILIES = {
    "multi_season_window": ["none", "career", "last_n", "recency_weighted"],
    "experience_feature": ["none", "age", "years_in_league", "career_games", "career_stage_bucket"],
    "sos_feature": ["none", "season_wide", "actual_games"],
    "include_depth_chart_competition": [True, False],
}

OUT_PATH = Path(__file__).resolve().parents[1] / "assets" / "feature_sweep_16.json"


def _summarize(report: dict) -> dict:
    return {
        "n_matched": report["n_matched"],
        "disagreement_edge": report["disagreement_edge"],
        "matched_population": report["matched_population"],
        "tier_accuracy": report["tier_accuracy"],
    }


def main() -> None:
    results = json.loads(OUT_PATH.read_text()) if OUT_PATH.exists() else {}

    for position in POSITION_CONFIGS:
        results.setdefault(position, {})
        for family, variants in FAMILIES.items():
            results[position].setdefault(family, {})
            for variant in variants:
                key = str(variant)
                if key in results[position][family]:
                    continue
                report = build_backtest_report(
                    position, seasons=STANDARD_BACKTEST_SEASONS, **{family: variant}
                )
                results[position][family][key] = _summarize(report)
                OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
                edge = results[position][family][key]["disagreement_edge"]
                print(f"{position:>3} {family:<32} {key:<20} corr={edge['correlation']:+.3f} "
                      f"ci={edge['correlation_ci']} win_rate={edge['win_rate']:.3f} "
                      f"n_large={edge['n_large_disagreements']}")


if __name__ == "__main__":
    main()
