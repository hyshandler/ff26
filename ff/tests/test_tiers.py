import pandas as pd
import pytest

from ff_model.tiers import derive_tiers, tier_accuracy_report


def test_derive_tiers_splits_on_largest_gaps() -> None:
    """Synthetic season: three tight clusters separated by two large gaps, and a
    target of 3 tiers -- the breaks must land exactly at the two big gaps, not at
    any of the small within-cluster gaps."""
    actual = pd.Series([100.0, 98.0, 96.0, 50.0, 48.0, 5.0, 4.0, 3.0])

    tiers = derive_tiers(actual, target_tiers=3)

    assert list(tiers) == [1, 1, 1, 2, 2, 3, 3, 3]


def test_derive_tiers_preserves_original_index() -> None:
    actual = pd.Series([10.0, 90.0, 50.0], index=["c", "a", "b"])

    tiers = derive_tiers(actual, target_tiers=3)

    assert tiers["a"] == 1
    assert tiers["b"] == 2
    assert tiers["c"] == 3


def test_derive_tiers_caps_at_available_players() -> None:
    actual = pd.Series([3.0, 2.0, 1.0])

    tiers = derive_tiers(actual, target_tiers=7)

    assert tiers.max() <= 3


def test_tier_accuracy_report_scores_model_and_adp_per_season() -> None:
    """Model rank exactly matches the actual order (perfect); ADP is fully reversed
    -- model must score perfect tier accuracy and ADP must score worse, giving a
    "beat" verdict. Actual points have real gap structure (three clusters of size
    4/3/3) so the tiering isn't degenerate under uniform spacing."""
    actual_points = [100.0, 99.0, 98.0, 97.0, 50.0, 49.0, 48.0, 10.0, 9.0, 8.0]
    rows = []
    for season in (2022, 2023):
        for position, points in enumerate(actual_points):
            rows.append(
                {
                    "target_season": season,
                    "actual_fantasy_points": points,
                    "full_projection_p50": points,
                    # Reversed draft order: the crowd ranked the worst actual
                    # player (last in the list) first.
                    "adp": float(len(actual_points) - position),
                }
            )
    df = pd.DataFrame(rows)

    report = tier_accuracy_report(
        df,
        actual_col="actual_fantasy_points",
        model_col="full_projection_p50",
        adp_col="adp",
        season_col="target_season",
        target_tiers=3,
    )

    assert report["model"]["accuracy"] == pytest.approx(1.0)
    assert report["model"]["off_by_one_rate"] == pytest.approx(0.0)
    assert report["model"]["gross_miss_rate"] == pytest.approx(0.0)
    assert report["adp"]["accuracy"] < 1.0
    # Gross misses (2+ tiers off) are tracked separately from off-by-one, per
    # ADR-0011/#14 -- not folded into a single "miss" bucket.
    assert report["adp"]["gross_miss_rate"] > 0.0
    assert report["verdict"] == "beat"
