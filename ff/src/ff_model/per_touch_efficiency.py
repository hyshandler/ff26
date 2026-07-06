import pandas as pd

from ff_model.feature_cache import cache_key, dataframe_fingerprint, disk_cached_frame

PER_TOUCH_EFFICIENCY_COLUMNS = [
    "prior_season_yards_per_target",
    "prior_season_yac_above_expectation",
]


def prior_season_yards_per_target(weekly_all_positions: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """[season, player_id, prior_season_yards_per_target] -- player's own immediately-
    preceding-season `receiving_yards / targets`, a skill-independent-of-volume usage-
    efficiency signal (issue #25 / ADR-0015). `season` follows `prior_season_totals`'s
    convention: a row for season S carries S-1's ratio. A player with zero prior-season
    targets (or no prior season at all) is absent from the result rather than given a
    divide-by-zero value.

    Disk-cached (see `feature_cache`) keyed on `weekly_all_positions`'s content and `seasons`.
    """
    fingerprint = dataframe_fingerprint(
        weekly_all_positions[["season", "player_id", "receiving_yards", "targets"]]
    )
    key = cache_key("prior_season_yards_per_target", fingerprint, {}, seasons=sorted(seasons))

    def compute() -> pd.DataFrame:
        totals = weekly_all_positions.groupby(["season", "player_id"])[
            ["receiving_yards", "targets"]
        ].sum()
        result = pd.DataFrame(
            {
                "season": totals.index.get_level_values("season") + 1,
                "player_id": totals.index.get_level_values("player_id"),
                "prior_season_yards_per_target": totals["receiving_yards"] / totals["targets"],
            }
        ).reset_index(drop=True)
        result = result.loc[totals["targets"].to_numpy() > 0]
        return result.loc[result["season"].isin(seasons)].reset_index(drop=True)

    return disk_cached_frame(key, compute)


def prior_season_yac_above_expectation_history(
    ngs_receiving: pd.DataFrame, seasons: list[int]
) -> pd.DataFrame:
    """[season, player_id, prior_season_yac_above_expectation] -- player's own immediately-
    preceding-season `avg_yac_above_expectation` from Next Gen Stats (issue #25). Same season
    convention as `prior_season_yards_per_target`: a row for season S carries S-1's value.

    NGS receiving data only exists from `nflverse.NGS_RECEIVING_EARLIEST_SEASON` (2016)
    onward (see `nflverse.load_ngs_receiving`) -- a target season S where S-1 predates that
    coverage simply produces no row here, left for LightGBM's native missingness handling to
    absorb (the same pattern already used for a player's first week of a season with no
    trailing average yet), rather than narrowing the training/backtest range around the gap.
    """
    shifted = ngs_receiving.rename(
        columns={"avg_yac_above_expectation": "prior_season_yac_above_expectation"}
    ).copy()
    shifted["season"] = shifted["season"] + 1
    return shifted.loc[shifted["season"].isin(seasons)].reset_index(drop=True)
