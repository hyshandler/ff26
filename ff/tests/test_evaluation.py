import pandas as pd

from ff_model.evaluation import mean_absolute_error


def test_mean_absolute_error_averages_the_absolute_differences() -> None:
    predicted = pd.Series([10.0, 20.0, 0.0])
    actual = pd.Series([12.0, 15.0, 5.0])

    # |10-12| + |20-15| + |0-5| = 2 + 5 + 5 = 12, / 3 = 4.0
    assert mean_absolute_error(predicted, actual) == 4.0
