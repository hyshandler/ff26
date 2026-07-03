from ff_model.backtest import walk_forward_splits


def test_slides_one_season_at_a_time_after_the_minimum_training_window() -> None:
    splits = walk_forward_splits(seasons=[2016, 2017, 2018, 2019], min_train_seasons=2)

    assert splits == [(2017, 2018), (2018, 2019)]


def test_raises_when_no_target_season_remains_after_the_minimum_training_window() -> None:
    import pytest

    with pytest.raises(ValueError):
        walk_forward_splits(seasons=[2016, 2017], min_train_seasons=2)


def test_never_predicts_a_season_using_data_from_that_season_or_later() -> None:
    splits = walk_forward_splits(seasons=[2016, 2017, 2018, 2019, 2020], min_train_seasons=3)

    for train_through_season, target_season in splits:
        assert target_season == train_through_season + 1
