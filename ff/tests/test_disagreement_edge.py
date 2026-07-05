import pandas as pd
import pytest

from ff_model.disagreement_edge import (
    LARGE_DISAGREEMENT_THRESHOLD,
    conditional_win_rate,
    encompassing_regression_beta,
    rank_residual_correlation,
)


def test_rank_residual_correlation_is_one_when_actual_matches_the_models_disagreement() -> None:
    """Per ADR-0014: `corr(model_rank - adp_rank, actual_rank - adp_rank)`. When the
    model's rank order exactly matches the actual outcome's rank order, the model's
    disagreements with ADP point exactly the right way -> perfect correlation."""
    df = pd.DataFrame(
        {
            # adp_rank (ascending, lower value = rank 1): A=1, B=2, C=3, D=4
            "adp": [10.0, 20.0, 30.0, 40.0],
            # model_rank (descending, higher value = rank 1): B=1, A=2, D=3, C=4
            "model": [30.0, 40.0, 10.0, 20.0],
            # actual_rank set identical to model_rank -> D_actual == D_model exactly.
            "actual": [30.0, 40.0, 10.0, 20.0],
        }
    )

    correlation = rank_residual_correlation(df, "model", "adp", "actual")

    assert correlation == pytest.approx(1.0)


def test_rank_residual_correlation_is_negative_when_actual_diverges_the_wrong_way() -> None:
    """When actual outcomes diverge from ADP in the opposite direction to the model's
    disagreement (the model's lone deviation from ADP -- swapping its rank-1 and
    rank-5 players -- points the wrong way), the correlation should come out negative,
    the sign ADR-0014 treats as evidence against the model's disagreements."""
    df = pd.DataFrame(
        {
            "adp": [1.0, 2.0, 3.0, 4.0, 5.0],
            # model swaps ADP's rank-1 and rank-5 players relative to adp_rank.
            "model": [5.0, 2.0, 3.0, 4.0, 1.0],
            "actual": [4.0, 5.0, 3.0, 1.0, 2.0],
        }
    )

    correlation = rank_residual_correlation(df, "model", "adp", "actual")

    assert correlation == pytest.approx(-0.7071067811865475)


def test_conditional_win_rate_all_wins_when_model_is_perfect_and_disagrees_with_adp() -> None:
    """Per ADR-0014: among large disagreements (threshold fixed in advance), the fraction
    where the model's rank landed closer to actual than ADP's did."""
    df = pd.DataFrame(
        {
            "adp": [10.0, 20.0, 30.0, 40.0, 50.0],  # adp_rank = 1,2,3,4,5
            # rank(ascending=False) of ascending values fully reverses the order:
            # model_rank = 5,4,3,2,1 -- the model disagrees with ADP on every player.
            "model": [10.0, 20.0, 30.0, 40.0, 50.0],
            "actual": [10.0, 20.0, 30.0, 40.0, 50.0],  # actual_rank == model_rank
        }
    )

    result = conditional_win_rate(df, "model", "adp", "actual", threshold=0)

    # The middle row has model_rank == adp_rank == 3 (D_model == 0), so it's excluded
    # from "large" disagreements at threshold=0; the other 4 all have model_rank ==
    # actual_rank exactly (model error 0) while ADP's error is nonzero -> model wins all.
    assert result["n_large_disagreements"] == 4
    assert result["win_rate"] == pytest.approx(1.0)


def test_conditional_win_rate_all_losses_when_actual_matches_adp_not_model() -> None:
    df = pd.DataFrame(
        {
            "adp": [10.0, 20.0, 30.0, 40.0, 50.0],
            "model": [10.0, 20.0, 30.0, 40.0, 50.0],  # model_rank = 5,4,3,2,1
            "actual": [50.0, 40.0, 30.0, 20.0, 10.0],  # actual_rank == adp_rank == 1,2,3,4,5
        }
    )

    result = conditional_win_rate(df, "model", "adp", "actual", threshold=0)

    assert result["n_large_disagreements"] == 4
    assert result["win_rate"] == pytest.approx(0.0)


def test_conditional_win_rate_excludes_disagreements_at_or_below_the_threshold() -> None:
    df = pd.DataFrame(
        {
            "adp": [10.0, 20.0, 30.0, 40.0, 50.0],
            "model": [10.0, 20.0, 30.0, 40.0, 50.0],  # model_rank = 5,4,3,2,1
            "actual": [10.0, 20.0, 30.0, 40.0, 50.0],  # actual_rank == model_rank
        }
    )

    # |model_rank - adp_rank| for these 5 rows is 4,2,0,2,4 -- a threshold of 3 keeps
    # only the two rows with a gap of 4.
    result = conditional_win_rate(df, "model", "adp", "actual", threshold=3)

    assert result["n_large_disagreements"] == 2


def test_rank_residual_correlation_ranks_within_season_when_season_col_given() -> None:
    """Draft rank only means something relative to players drafted the same season --
    pooling ranks across seasons would compare players who were never actually drafted
    against each other. Two independent, identically-shaped seasons stacked together
    should score the same as either season alone once ranks are computed per season."""
    one_season = pd.DataFrame(
        {
            "adp": [10.0, 20.0, 30.0, 40.0, 50.0],
            "model": [30.0, 40.0, 10.0, 20.0, 50.0],
            "actual": [30.0, 40.0, 10.0, 20.0, 50.0],
            "season": [2022] * 5,
        }
    )
    two_seasons = pd.concat([one_season, one_season.assign(season=2023)], ignore_index=True)

    correlation = rank_residual_correlation(two_seasons, "model", "adp", "actual", season_col="season")

    assert correlation == pytest.approx(rank_residual_correlation(one_season, "model", "adp", "actual"))


def test_large_disagreement_threshold_is_one_draft_round() -> None:
    """Fixed in advance per ADR-0014/research doc -- not tuned post-hoc."""
    assert LARGE_DISAGREEMENT_THRESHOLD == 12


def test_encompassing_regression_beta_recovers_a_pure_model_effect() -> None:
    """Per ADR-0014: `actual ~ b0 + b1*adp + b2*model` -- when actual is an exact linear
    function of model alone (no adp dependence), beta2 should recover that coefficient."""
    df = pd.DataFrame(
        {
            "adp": [1.0, 2.0, 3.0, 4.0, 5.0],
            "model": [5.0, 3.0, 4.0, 1.0, 2.0],
            "actual": [10.0, 6.0, 8.0, 2.0, 4.0],  # exactly 2 * model, independent of adp
        }
    )

    beta2 = encompassing_regression_beta(df, "adp", "model", "actual")

    assert beta2 == pytest.approx(2.0)
