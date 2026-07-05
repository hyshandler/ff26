import numpy as np
import pandas as pd

LARGE_DISAGREEMENT_THRESHOLD = 12
"""One full draft round in a 12-team league, fixed in advance per ADR-0014 and
`docs/research/disagreement-edge-metric.md` -- a "meaningful disagreement" cutoff chosen
after seeing results would be overfitting, so this is picked from draft-actionability
reasoning, not tuned to maximize the win rate."""


def _ranks(
    df: pd.DataFrame, model_col: str, adp_col: str, actual_col: str, season_col: str | None = None
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rank-1-is-best ranks for each column: ADP is already a draft-order rank (lower
    is better), while `model_col`/`actual_col` are point projections/totals (higher is
    better), so only those two need `ascending=False`.

    When `season_col` is given, ranks are computed within each season group rather than
    across the whole frame -- draft rank only means something relative to the other
    players drafted the same season, so pooling ranks across multiple backtest seasons
    would compare players who were never actually drafted against each other.
    """
    if season_col is not None:
        adp_rank = df.groupby(season_col)[adp_col].rank()
        model_rank = df.groupby(season_col)[model_col].rank(ascending=False)
        actual_rank = df.groupby(season_col)[actual_col].rank(ascending=False)
    else:
        adp_rank = df[adp_col].rank()
        model_rank = df[model_col].rank(ascending=False)
        actual_rank = df[actual_col].rank(ascending=False)
    return model_rank, adp_rank, actual_rank


def rank_residual_correlation(
    df: pd.DataFrame, model_col: str, adp_col: str, actual_col: str, season_col: str | None = None
) -> float:
    """Disagreement Edge's headline number (ADR-0014):
    `corr(model_rank - adp_rank, actual_rank - adp_rank)`.

    Positive means that when the model diverges from ADP, actual outcomes tend to
    diverge from ADP in the same direction -- i.e. the model's disagreements with the
    room carry real information, not noise. Pass `season_col` when `df` spans multiple
    backtest seasons so ranks are computed within each season (see `_ranks`).
    """
    model_rank, adp_rank, actual_rank = _ranks(df, model_col, adp_col, actual_col, season_col)
    d_model = model_rank - adp_rank
    d_actual = actual_rank - adp_rank
    return float(d_model.corr(d_actual))


def conditional_win_rate(
    df: pd.DataFrame,
    model_col: str,
    adp_col: str,
    actual_col: str,
    threshold: int = LARGE_DISAGREEMENT_THRESHOLD,
    season_col: str | None = None,
) -> dict:
    """Disagreement Edge's decision verdict (ADR-0014): among player-seasons where the
    model disagrees with ADP by more than `threshold` rank positions, the fraction where
    the model's rank landed closer to the actual outcome than ADP's did.

    Unlike tier accuracy, this isn't a beat/tie/lose contest with ADP as a whole -- ADP
    can't disagree with itself, so there's no ADP-vs-ADP baseline to compare against. The
    null hypothesis this win rate is judged against is 50% (pure noise), not "beats ADP's
    own win rate." Pass `season_col` when `df` spans multiple backtest seasons so ranks
    are computed within each season (see `_ranks`).
    """
    model_rank, adp_rank, actual_rank = _ranks(df, model_col, adp_col, actual_col, season_col)
    large = (model_rank - adp_rank).abs() > threshold
    n_large = int(large.sum())
    if n_large == 0:
        return {"n_large_disagreements": 0, "win_rate": float("nan")}

    model_error = (actual_rank - model_rank).abs()[large]
    adp_error = (actual_rank - adp_rank).abs()[large]
    return {"n_large_disagreements": n_large, "win_rate": float((model_error < adp_error).mean())}


def encompassing_regression_beta(df: pd.DataFrame, adp_col: str, model_col: str, actual_col: str) -> float:
    """Disagreement Edge's secondary, development-only check (ADR-0014):
    `actual ~ b0 + b1*adp + b2*model`. A beta2 distinguishable from zero means the model
    adds real information beyond ADP alone, confirming `rank_residual_correlation` isn't
    an artifact of a confound rather than genuine model skill.
    """
    design = np.column_stack(
        [
            np.ones(len(df)),
            df[adp_col].to_numpy(dtype=float),
            df[model_col].to_numpy(dtype=float),
        ]
    )
    outcome = df[actual_col].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    return float(beta[2])
