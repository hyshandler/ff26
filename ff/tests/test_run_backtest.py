import pandas as pd
import pytest

from ff_model.run_backtest import run_backtest


@pytest.mark.network
def test_concatenates_every_split_into_one_dataframe_with_a_split_marker() -> None:
    result = run_backtest("RB", seasons=[2020, 2021, 2022, 2023], min_train_seasons=2)

    assert isinstance(result, pd.DataFrame)
    assert set(result["target_season"]) == {2022, 2023}

    # Each split's rows are traceable back to the season boundary that produced them.
    by_split = result.groupby("target_season")["train_through_season"].unique()
    assert by_split[2022].tolist() == [2021]
    assert by_split[2023].tolist() == [2022]
