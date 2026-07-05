import pandas as pd

from ff_model.disagreement_edge import conditional_win_rate, rank_residual_correlation
from ff_model.evaluation import (
    matched_population_report,
    mean_absolute_error,
    season_block_bootstrap_interval,
)
from ff_model.tiers import tier_accuracy_report


def build_position_report(df: pd.DataFrame) -> dict:
    """Backtest report for one position, leading with the Matched Population per ADR-0010.

    `df` is a `run_backtest` result joined with `with_actual_outcomes`,
    `with_adp_benchmark`, and `with_naive_baseline`: it must carry
    `full_projection_p50`, `naive_full_projection`, `adp`, `actual_fantasy_points`,
    and `target_season`. The Matched Population is every row with a non-null `adp`
    -- the players the crowd actually drafted (per ADR-0010's Matched Population);
    rows outside it (undraftable depth players) would trivially inflate
    full-population rho.

    ADP ranks lower-is-better, unlike `full_projection_p50`/`naive_full_projection`/
    `actual_fantasy_points`, so it's negated here before ranking -- callers don't
    need to pre-negate it themselves.
    """
    df = df.assign(_neg_adp=-df["adp"])
    matched_mask = df["adp"].notna()

    report = matched_population_report(
        df,
        actual_column="actual_fantasy_points",
        matched_mask=matched_mask,
        prediction_columns={
            "model": "full_projection_p50",
            "adp": "_neg_adp",
            "naive": "naive_full_projection",
        },
    )
    report["n"] = len(df)

    # MAE computed on the same Matched Population as the rho numbers above -- the
    # undraftable depth players ADR-0010 excludes from rho would distort MAE just
    # as easily (e.g. a wildly-off projection on a bench player nobody drafted).
    matched = df.loc[matched_mask]
    report["model_mae"] = mean_absolute_error(matched["full_projection_p50"], matched["actual_fantasy_points"])
    report["naive_mae"] = mean_absolute_error(matched["naive_full_projection"], matched["actual_fantasy_points"])

    # Disagreement Edge (ADR-0014) -- the v2 headline verdict, on the same Matched
    # Population as everything else above. CIs resample whole `target_season` blocks
    # rather than rows, since the season is the real unit of independence.
    win_rate_result = conditional_win_rate(
        matched, "full_projection_p50", "adp", "actual_fantasy_points", season_col="target_season"
    )
    report["disagreement_edge"] = {
        "correlation": rank_residual_correlation(
            matched, "full_projection_p50", "adp", "actual_fantasy_points", season_col="target_season"
        ),
        "correlation_ci": season_block_bootstrap_interval(
            matched,
            "target_season",
            lambda block: rank_residual_correlation(
                block, "full_projection_p50", "adp", "actual_fantasy_points", season_col="target_season"
            ),
        ),
        "win_rate": win_rate_result["win_rate"],
        "n_large_disagreements": win_rate_result["n_large_disagreements"],
        "win_rate_ci": season_block_bootstrap_interval(
            matched,
            "target_season",
            lambda block: conditional_win_rate(
                block, "full_projection_p50", "adp", "actual_fantasy_points", season_col="target_season"
            )["win_rate"],
        ),
    }

    # Tier accuracy (ADR-0011) -- supporting signal alongside Disagreement Edge,
    # also on the Matched Population: how often each ranking places a player in
    # his true tier, derived per-season from actual outcomes.
    report["tier_accuracy"] = tier_accuracy_report(
        matched,
        actual_col="actual_fantasy_points",
        model_col="full_projection_p50",
        adp_col="adp",
        season_col="target_season",
    )
    return report
