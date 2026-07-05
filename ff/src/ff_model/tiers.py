import numpy as np
import pandas as pd


def derive_tiers(actual: pd.Series, target_tiers: int = 7) -> pd.Series:
    """Cluster players into tiers (1 = best) by the largest gaps in sorted `actual`
    outcomes, per ADR-0011: tiers track real breaks in the season's own point
    distribution rather than fixed-size buckets, which would just re-derive ADP's
    own round structure.

    Picks the `target_tiers - 1` largest consecutive-gap positions as tier
    boundaries. If fewer than `target_tiers` players are available, every player
    gets their own tier rather than erroring.
    """
    n = len(actual)
    if n == 0:
        return pd.Series([], index=actual.index, dtype=int)

    sorted_actual = actual.sort_values(ascending=False)
    n_breaks = min(target_tiers - 1, n - 1)

    if n_breaks <= 0:
        tier_by_sorted_position = np.ones(n, dtype=int)
    else:
        gaps = sorted_actual.to_numpy()[:-1] - sorted_actual.to_numpy()[1:]
        break_positions = set(np.argsort(gaps, kind="stable")[-n_breaks:].tolist())
        tier_by_sorted_position = np.empty(n, dtype=int)
        tier = 1
        for i in range(n):
            tier_by_sorted_position[i] = tier
            if i in break_positions:
                tier += 1

    tiers = pd.Series(tier_by_sorted_position, index=sorted_actual.index)
    return tiers.reindex(actual.index)


def _assign_tiers_by_rank(rank: pd.Series, tier_sizes: pd.Series) -> pd.Series:
    """Maps a 1-is-best `rank` column onto the tier structure described by
    `tier_sizes` (tier number -> player count, ordered best to worst) -- used to
    score a rank-only predictor (no point magnitudes of its own, e.g. ADP) against
    tiers derived from actual points: the predictor's top `tier_sizes[1]` players by
    rank are assigned tier 1, the next `tier_sizes[2]` are tier 2, and so on.
    """
    sorted_sizes = tier_sizes.sort_index()
    boundaries = np.cumsum(sorted_sizes.to_numpy())
    tier_labels = sorted_sizes.index.to_numpy()
    predicted = np.searchsorted(boundaries, rank.to_numpy(), side="left")
    predicted = np.clip(predicted, 0, len(tier_labels) - 1)
    return pd.Series(tier_labels[predicted], index=rank.index)


def tier_accuracy_report(
    df: pd.DataFrame,
    actual_col: str,
    model_col: str,
    adp_col: str,
    season_col: str,
    target_tiers: int = 7,
) -> dict:
    """Tier accuracy and off-by-one rate for the model and ADP, per ADR-0011:
    true tiers are derived from each season's actual outcomes, then both the
    model's and ADP's rank orders are sliced into groups of those same tier sizes
    to see how often each places a player in his true tier.

    `df` should already be restricted to the Matched Population (per ADR-0010).
    `model_col` is higher-is-better (a projection); `adp_col` is lower-is-better
    (a draft-order rank), consistent with the rest of the Disagreement Edge/backtest
    report machinery.
    """
    true_tiers = []
    model_tiers = []
    adp_tiers = []

    for _, season_df in df.groupby(season_col):
        actual = season_df[actual_col]
        season_true_tiers = derive_tiers(actual, target_tiers=target_tiers)
        tier_sizes = season_true_tiers.value_counts()

        model_rank = season_df[model_col].rank(ascending=False)
        adp_rank = season_df[adp_col].rank(ascending=True)

        true_tiers.append(season_true_tiers)
        model_tiers.append(_assign_tiers_by_rank(model_rank, tier_sizes))
        adp_tiers.append(_assign_tiers_by_rank(adp_rank, tier_sizes))

    true_tiers = pd.concat(true_tiers)
    model_tiers = pd.concat(model_tiers)
    adp_tiers = pd.concat(adp_tiers)

    def _score(predicted_tiers: pd.Series) -> dict:
        diff = (predicted_tiers - true_tiers).abs()
        return {
            "accuracy": float((diff == 0).mean()),
            "off_by_one_rate": float((diff == 1).mean()),
            # Tracked separately from off-by-one per ADR-0011/#14: a miss of two or
            # more tiers is a much worse signal than a near-boundary miss.
            "gross_miss_rate": float((diff > 1).mean()),
        }

    model_score = _score(model_tiers)
    adp_score = _score(adp_tiers)

    if model_score["accuracy"] > adp_score["accuracy"]:
        verdict = "beat"
    elif model_score["accuracy"] < adp_score["accuracy"]:
        verdict = "lost"
    else:
        verdict = "tied"

    return {"model": model_score, "adp": adp_score, "verdict": verdict}
