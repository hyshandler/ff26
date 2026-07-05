import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd

_DISK_CACHE_DIR = Path(__file__).resolve().parents[2] / "assets" / "feature_cache"
"""On-disk cache for engineered feature tables (trailing averages/shares, multi-season
memory windows). Each cached function is pure given its inputs, and the underlying
nflverse data is itself frozen once fetched (see `nflverse._disk_cached`), so a feature
table computed for a given key never goes stale. Lives under the repo's `assets/` dir so
it's computed once per checkout and reused across walk-forward splits and repeated
backtest/sweep runs, not recomputed from scratch on every call."""


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Cheap content hash of a dataframe's values, order-independent in row order.

    Used as part of a cache key instead of e.g. just the season range present in `df` --
    real nflverse data for a season is immutable so this resolves to the same value across
    repeated production calls, but it also keeps the cache correct when two calls share the
    same season numbers with different underlying rows (e.g. distinct unit-test fixtures).
    """
    hashed = pd.util.hash_pandas_object(df.sort_values(list(df.columns)).reset_index(drop=True), index=False)
    return hashlib.sha1(hashed.to_numpy().tobytes()).hexdigest()[:16]


def cache_key(name: str, frame_fingerprint: str, stat_columns: dict[str, str], **params: object) -> str:
    """Stable hash-based key for a feature function's inputs.

    `frame_fingerprint` should come from `dataframe_fingerprint` over just the columns the
    function actually reads, so unrelated columns/rows outside that subset don't force a
    cache miss.
    """
    payload = {
        "name": name,
        "frame_fingerprint": frame_fingerprint,
        "stat_columns": sorted(stat_columns.items()),
        "params": sorted(params.items()),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return f"{name}_{digest}"


def disk_cached_frame(key: str, compute: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Compute-once-then-reuse-forever cache, mirroring `nflverse._disk_cached`."""
    path = _DISK_CACHE_DIR / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df
