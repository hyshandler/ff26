import pandas as pd

AGE_BUCKET_EDGES = [0, 24, 27, 30, 100]
AGE_BUCKET_LABELS = ["<24", "24-26", "27-29", "30+"]


def age_bucket(age: pd.Series) -> pd.Series:
    """Buckets `age` into career-stage cohorts, shared between the Games-Played Estimate
    heuristic (`games_played.py`) and the career-stage experience feature
    (`experience_features.py`) so the two never silently drift apart."""
    return pd.cut(age, bins=AGE_BUCKET_EDGES, labels=AGE_BUCKET_LABELS, right=False)
