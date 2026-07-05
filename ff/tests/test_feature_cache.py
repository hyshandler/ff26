import subprocess
import sys
import textwrap

import pandas as pd

from ff_model.feature_cache import cache_key, dataframe_fingerprint, disk_cached_frame


def test_disk_cached_frame_returns_cached_value_without_recomputing(tmp_path, monkeypatch) -> None:
    import ff_model.feature_cache as feature_cache

    monkeypatch.setattr(feature_cache, "_DISK_CACHE_DIR", tmp_path)
    calls = []

    def compute() -> pd.DataFrame:
        calls.append(1)
        return pd.DataFrame({"a": [1, 2, 3]})

    key = "some_key"
    first = disk_cached_frame(key, compute)
    second = disk_cached_frame(key, compute)

    pd.testing.assert_frame_equal(first, second)
    assert len(calls) == 1  # second call hit the disk cache, never called compute again


def test_cache_key_differs_when_frame_fingerprint_differs() -> None:
    stat_columns = {"trailing_avg_carries": "carries"}
    key_a = cache_key("trailing_player_averages", "fingerprint-a", stat_columns)
    key_b = cache_key("trailing_player_averages", "fingerprint-b", stat_columns)
    assert key_a != key_b


def test_dataframe_fingerprint_differs_for_different_content() -> None:
    df_a = pd.DataFrame({"season": [2022], "player_id": ["p1"], "carries": [10]})
    df_b = pd.DataFrame({"season": [2022], "player_id": ["p1"], "carries": [99]})
    assert dataframe_fingerprint(df_a) != dataframe_fingerprint(df_b)


def test_dataframe_fingerprint_is_blind_to_row_order() -> None:
    df = pd.DataFrame({"season": [2022, 2023], "player_id": ["p1", "p2"], "carries": [10, 20]})
    fingerprint_forward = dataframe_fingerprint(df)
    fingerprint_reversed = dataframe_fingerprint(df.iloc[::-1])
    assert fingerprint_forward == fingerprint_reversed


def test_fingerprint_matches_across_separate_processes_regardless_of_hash_seed() -> None:
    """End-to-end regression check for the same bug: run the real fingerprint-producing
    column selection (`sorted(set(stat_columns.values()))`, as used in `features.py`) in two
    subprocesses with different `PYTHONHASHSEED`s and confirm they agree.
    """
    script = textwrap.dedent(
        """
        import pandas as pd
        from ff_model.feature_cache import dataframe_fingerprint

        stat_columns = {"trailing_avg_carries": "carries", "trailing_avg_targets": "targets"}
        df = pd.DataFrame({
            "season": [2022, 2022],
            "player_id": ["p1", "p2"],
            "carries": [10, 5],
            "targets": [2, 8],
        })
        columns = ["season", "player_id", *sorted(set(stat_columns.values()))]
        print(dataframe_fingerprint(df[columns]))
        """
    )
    fingerprints = set()
    for seed in ("1", "999"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": __import__("os").environ["PATH"]},
            check=True,
        )
        fingerprints.add(result.stdout.strip())
    assert len(fingerprints) == 1
