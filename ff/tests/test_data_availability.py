import pytest

from ff_model.data_availability import assert_no_participation_sourced_features


def test_allows_raw_box_score_columns() -> None:
    assert_no_participation_sourced_features(["carries", "rushing_yards", "receptions"])


def test_rejects_a_participation_or_ftn_sourced_column() -> None:
    with pytest.raises(ValueError):
        assert_no_participation_sourced_features(["target_share", "offense_snaps"])
