import pandas as pd

from ff_model.age_buckets import age_bucket


def test_age_bucket_assigns_the_correct_cohort_at_each_edge() -> None:
    ages = pd.Series([23.0, 24.0, 26.9, 27.0, 29.9, 30.0])

    result = age_bucket(ages)

    assert list(result) == ["<24", "24-26", "24-26", "27-29", "27-29", "30+"]
