from functools import lru_cache
from pathlib import Path
from typing import Callable

import appdirs
import nfl_data_py as nfl
import pandas as pd


_DISK_CACHE_DIR = Path(__file__).resolve().parents[2] / "assets" / "nflverse_cache"
"""Root for this module's own on-disk parquet cache -- distinct from `nfl_data_py`'s
own cache dir (used by `_ensure_pbp_cached`), since every loader here besides play-by-play
fetches straight from the network on every call otherwise. Lives under the repo's `assets/`
dir (not a user-level cache dir) so the download only ever has to happen once per checkout,
not once per machine."""


def _disk_cached(relative_path: str, fetch: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    """Fetch-once-then-reuse-forever cache: nflverse's release files for a given season
    are immutable once published, so a local parquet copy never goes stale and lets every
    loader below survive across process restarts, not just within one Walk-Forward run."""
    path = _DISK_CACHE_DIR / relative_path
    if path.exists():
        return pd.read_parquet(path)
    df = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


FANTASY_REGULAR_SEASON_MAX_WEEK = 17
"""Standard fantasy leagues play a REG-season-only schedule capped at week 17,
regardless of era: 2021+ seasons have an 18th REG week in nflverse's raw data
(added for scheduling reasons when the NFL expanded to 17 games), but fantasy
leagues didn't adopt it as a scored week."""

_STATS_PLAYER_WEEKLY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{0}.parquet"
)
"""nflverse's current per-player-week release. Supersedes `nfl_data_py.import_weekly_data`'s
hardcoded `player_stats` release, which is frozen at the 2024 season and 404s for 2025+."""

_WEEKLY_STATS_COLUMN_RENAMES = {
    "team": "recent_team",
    "passing_interceptions": "interceptions",
}
"""`stats_player` renamed these relative to the retired `player_stats` release; renamed back
here so every downstream feature/config module keeps using the names it already knows."""


def _fetch_weekly_stats_for_season(season: int) -> pd.DataFrame:
    df = pd.read_parquet(_STATS_PLAYER_WEEKLY_URL.format(season))
    df = df.rename(columns=_WEEKLY_STATS_COLUMN_RENAMES)
    return df.loc[(df["season_type"] == "REG") & (df["week"] <= FANTASY_REGULAR_SEASON_MAX_WEEK)]


@lru_cache(maxsize=None)
def _weekly_stats_for_season(season: int) -> pd.DataFrame:
    return _disk_cached(f"weekly_stats/{season}.parquet", lambda: _fetch_weekly_stats_for_season(season))


def load_weekly_stats(seasons: list[int]) -> pd.DataFrame:
    """Raw weekly box-score stats from nflverse, at the game/week grain.

    Filtered to the regular season through `FANTASY_REGULAR_SEASON_MAX_WEEK`:
    postseason games and any week beyond that aren't part of a standard fantasy
    league's scored schedule.

    nflverse publishes this one parquet file per season, so each season's fetch is
    cached in-process the first time it's requested -- the Walk-Forward Backtest's
    expanding window means most seasons get requested again by every later split, and
    without this, that's a full re-download of every prior season on every split.
    """
    return pd.concat([_weekly_stats_for_season(season) for season in seasons])


@lru_cache(maxsize=None)
def _all_schedules() -> pd.DataFrame:
    """nflverse's schedule source is one fixed file covering every season (unlike the
    per-season files above), so this is cached once, unfiltered, rather than re-fetched
    per call and filtered locally by `load_schedules`."""
    return _disk_cached("schedules.parquet", lambda: nfl.import_schedules(range(1999, 2030)))


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    """Team schedules from nflverse, long-format one row per (season, week, team) with
    that week's opponent -- unpivoted from home/away, REG season only through
    `FANTASY_REGULAR_SEASON_MAX_WEEK` (matching `load_weekly_stats`), for Strength of
    Schedule features. A team's bye week has no row, same as any other data source here.
    """
    df = _all_schedules()
    regular_season = df.loc[
        df["season"].isin(seasons)
        & (df["game_type"] == "REG")
        & (df["week"] <= FANTASY_REGULAR_SEASON_MAX_WEEK)
    ]
    home = regular_season[["season", "week", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent_team"}
    )
    away = regular_season[["season", "week", "home_team", "away_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent_team"}
    )
    return pd.concat([home, away], ignore_index=True)


def load_team_scores(seasons: list[int]) -> pd.DataFrame:
    """Team scoring from nflverse, long-format one row per (season, week, team) with
    that week's own points scored -- unpivoted from home/away like `load_schedules`,
    REG season only through `FANTASY_REGULAR_SEASON_MAX_WEEK`, for the Own-Team
    Offensive Environment feature (issue #24). A team's bye week has no row.
    """
    df = _all_schedules()
    regular_season = df.loc[
        df["season"].isin(seasons)
        & (df["game_type"] == "REG")
        & (df["week"] <= FANTASY_REGULAR_SEASON_MAX_WEEK)
    ]
    home = regular_season[["season", "week", "home_team", "home_score"]].rename(
        columns={"home_team": "team", "home_score": "points"}
    )
    away = regular_season[["season", "week", "away_team", "away_score"]].rename(
        columns={"away_team": "team", "away_score": "points"}
    )
    return pd.concat([home, away], ignore_index=True)


@lru_cache(maxsize=None)
def _seasonal_roster_for_season(season: int) -> pd.DataFrame:
    return _disk_cached(
        f"seasonal_rosters/{season}.parquet", lambda: nfl.import_seasonal_rosters([season])
    )


def load_seasonal_rosters(seasons: list[int]) -> pd.DataFrame:
    """Seasonal rosters from nflverse, carrying each player's rookie_year and pfr_id.

    One parquet file per season (like `load_weekly_stats`); same per-season caching
    to avoid re-fetching seasons every later Walk-Forward split already requested.
    """
    return pd.concat([_seasonal_roster_for_season(season) for season in seasons])


def pfr_id_crosswalk(rosters: pd.DataFrame) -> pd.Series:
    """Map nflverse's gsis-style player_id to the pfr_id snap counts are keyed by."""
    return (
        rosters.dropna(subset=["pfr_id"])
        .drop_duplicates("player_id")
        .set_index("player_id")["pfr_id"]
    )


_PBP_CACHE_DIR = _DISK_CACHE_DIR / "pbp"


def _ensure_pbp_cached(seasons: list[int]) -> None:
    """Play-by-play is nfl_data_py's heaviest download by far, and `import_pbp_data`
    re-fetches it from GitHub every call unless told to use its local disk cache --
    this populates that cache once per season instead of on every call. Uses `alt_path`
    to keep this cache under the repo's `assets/` dir alongside every other loader here,
    rather than `nfl_data_py`'s own OS-level cache dir."""
    missing = [s for s in seasons if not (_PBP_CACHE_DIR / f"season={s}").is_dir()]
    if missing:
        nfl.cache_pbp(missing, alt_path=str(_PBP_CACHE_DIR))


@lru_cache(maxsize=None)
def _red_zone_rush_attempts_for_season(season: int) -> pd.DataFrame:
    _ensure_pbp_cached([season])
    pbp: pd.DataFrame = nfl.import_pbp_data(
        [season], downcast=True, cache=True, alt_path=str(_PBP_CACHE_DIR)
    )
    red_zone_rushes = pbp.loc[
        (pbp["rush_attempt"] == 1) & (pbp["yardline_100"] <= 20) & pbp["rusher_player_id"].notna()
    ]
    return (
        red_zone_rushes.groupby(["season", "week", "rusher_player_id"])
        .size()
        .rename("red_zone_carries")
        .reset_index()
        .rename(columns={"rusher_player_id": "player_id"})
    )


def load_red_zone_rush_attempts(seasons: list[int]) -> pd.DataFrame:
    """Per player-week red-zone (inside the 20) rush attempts, from nflverse play-by-play.

    Play-by-play is nflverse's heaviest per-season load by far, so this memoizes in-process
    per season (like every other loader here) -- without it, the Walk-Forward Backtest's
    expanding window re-decodes every prior season's full play-by-play on every split.
    """
    return pd.concat(
        [_red_zone_rush_attempts_for_season(season) for season in seasons], ignore_index=True
    )


INJURY_REPORTS_EARLIEST_SEASON = 2009
"""nflverse's injury report data doesn't go back further than this."""


def _fetch_injury_reports_for_season(season: int) -> pd.DataFrame:
    df: pd.DataFrame = nfl.import_injuries([season])
    regular_season = df.loc[df["game_type"] == "REG"]
    return regular_season.rename(columns={"gsis_id": "player_id"})[
        ["player_id", "season", "week", "report_status"]
    ]


@lru_cache(maxsize=None)
def _injury_reports_for_season(season: int) -> pd.DataFrame:
    return _disk_cached(
        f"injury_reports/{season}.parquet", lambda: _fetch_injury_reports_for_season(season)
    )


def load_injury_reports(seasons: list[int]) -> pd.DataFrame:
    """Weekly injury report status per player, regular season only.

    `gsis_id` already matches the gsis-style `player_id` used everywhere else
    in this pipeline, so no crosswalk is needed (unlike snap counts' pfr_id).
    Seasons before `INJURY_REPORTS_EARLIEST_SEASON` are silently dropped, since
    nflverse has no injury-report data for them at all. One parquet file per season
    (like `load_weekly_stats`); same per-season caching applies.
    """
    available_seasons = [s for s in seasons if s >= INJURY_REPORTS_EARLIEST_SEASON]
    if not available_seasons:
        return pd.DataFrame(columns=["player_id", "season", "week", "report_status"])

    return pd.concat([_injury_reports_for_season(season) for season in available_seasons])


@lru_cache(maxsize=None)
def _all_draft_picks() -> pd.DataFrame:
    """nflverse's draft-picks source is one fixed file covering every season (unlike the
    per-season files above), so this is cached once, unfiltered, rather than re-fetched
    per call and filtered locally by `load_draft_picks`."""
    return _disk_cached("draft_picks.parquet", nfl.import_draft_picks)


def load_draft_picks(seasons: list[int]) -> pd.DataFrame:
    """Draft picks per season, for the Depth-Chart Competition Feature's rookie-arrival signal.

    `gsis_id` already matches the gsis-style `player_id` used everywhere else.
    """
    df = _all_draft_picks()
    df = df.loc[df["season"].isin(seasons)]
    return df.rename(columns={"gsis_id": "player_id"})[
        ["player_id", "season", "round", "position", "team"]
    ]


SNAP_COUNTS_EARLIEST_SEASON = 2012
"""nflverse's Pro-Football-Reference-sourced snap counts don't go back further than this."""


@lru_cache(maxsize=None)
def _snap_counts_for_season(season: int) -> pd.DataFrame:
    return _disk_cached(f"snap_counts/{season}.parquet", lambda: nfl.import_snap_counts([season]))


def load_offense_snap_pct(seasons: list[int], pfr_id_by_player_id: pd.Series) -> pd.DataFrame:
    """Per player-week share of offensive snaps played, from nflverse (Pro Football Reference).

    `pfr_id_by_player_id` crosswalks nflverse's gsis-style player_id (used everywhere else
    in this pipeline) to the pfr_player_id snap counts are keyed by — typically the
    `pfr_id` column from `load_seasonal_rosters`. Seasons before
    `SNAP_COUNTS_EARLIEST_SEASON` are silently dropped, since nflverse has no snap-count
    data for them at all; earlier player-weeks simply carry a missing snap_pct feature.
    One parquet file per season (like `load_weekly_stats`); same per-season caching
    applies to the raw fetch, before the (call-specific) crosswalk join.
    """
    available_seasons = [s for s in seasons if s >= SNAP_COUNTS_EARLIEST_SEASON]
    if not available_seasons:
        return pd.DataFrame(columns=["player_id", "season", "week", "offense_pct"])

    snap_counts = pd.concat([_snap_counts_for_season(season) for season in available_seasons])
    crosswalk = pfr_id_by_player_id.drop_duplicates()
    player_id_by_pfr_id = pd.Series(crosswalk.index, index=crosswalk.values)
    snap_counts["player_id"] = snap_counts["pfr_player_id"].map(player_id_by_pfr_id)
    return snap_counts.loc[
        snap_counts["player_id"].notna(), ["player_id", "season", "week", "offense_pct"]
    ]


NGS_RECEIVING_EARLIEST_SEASON = 2016
"""nflverse's Next Gen Stats receiving data doesn't go back further than this."""


@lru_cache(maxsize=None)
def _ngs_receiving_for_season(season: int) -> pd.DataFrame:
    return _disk_cached(
        f"ngs_receiving/{season}.parquet", lambda: nfl.import_ngs_data("receiving", [season])
    )


def load_ngs_receiving(seasons: list[int]) -> pd.DataFrame:
    """Season-level Next Gen Stats receiving efficiency per player, from nflverse.

    Exposes `avg_yac_above_expectation` for the Per-Touch Efficiency feature (issue #25).
    Seasons before `NGS_RECEIVING_EARLIEST_SEASON` are silently dropped, same gap-handling
    pattern as `load_offense_snap_pct`'s `SNAP_COUNTS_EARLIEST_SEASON` -- nflverse simply has
    no NGS data for them. `nfl_data_py.import_ngs_data` returns one row per player per week
    plus a `week == 0` row holding that player's season-level aggregate; filtered to those
    `week == 0` rows and `season_type == "REG"`, matching `FANTASY_REGULAR_SEASON_MAX_WEEK`'s
    regular-season-only convention elsewhere. One parquet file per season (like
    `load_weekly_stats`); same per-season caching applies.
    """
    available_seasons = [s for s in seasons if s >= NGS_RECEIVING_EARLIEST_SEASON]
    if not available_seasons:
        # Explicit dtypes, not just column names: an untyped empty frame defaults
        # every column to `object`, which then poisons `avg_yac_above_expectation`
        # as `object` all the way through the merge chain into LightGBM's training
        # matrix -- surfaces for real on any Walk-Forward split trained entirely on
        # pre-2016 seasons, not just in synthetic test data.
        return pd.DataFrame(
            {
                "player_id": pd.Series(dtype="object"),
                "season": pd.Series(dtype="int64"),
                "avg_yac_above_expectation": pd.Series(dtype="float64"),
            }
        )

    ngs = pd.concat([_ngs_receiving_for_season(season) for season in available_seasons])
    season_level = ngs.loc[(ngs["week"] == 0) & (ngs["season_type"] == "REG")]
    return season_level.rename(columns={"player_gsis_id": "player_id"})[
        ["player_id", "season", "avg_yac_above_expectation"]
    ]
