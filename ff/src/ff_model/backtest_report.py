import pandas as pd

from ff_model.evaluation import matched_population_report, mean_absolute_error


def build_position_report(df: pd.DataFrame) -> dict:
    """Backtest report for one position, leading with the Matched Population per ADR-0010.

    `df` is a `run_backtest` result joined with `with_actual_outcomes`,
    `with_adp_benchmark`, and `with_naive_baseline`: it must carry
    `full_projection_p50`, `naive_full_projection`, `adp`, and `actual_fantasy_points`.
    The Matched Population is every row with a non-null `adp` -- the players the
    crowd actually drafted (per ADR-0010's Matched Population); rows outside it
    (undraftable depth players) would trivially inflate full-population rho.

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
    return report
