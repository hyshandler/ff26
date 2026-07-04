from functools import lru_cache
from pathlib import Path

import appdirs
import nfl_data_py as nfl
import pandas as pd


FANTASY_REGULAR_SEASON_MAX_WEEK = 17
"""Standard fantasy leagues play a REG-season-only schedule capped at week 17,
regardless of era: 2021+ seasons have an 18th REG week in nflverse's raw data
(added for scheduling reasons when the NFL expanded to 17 games), but fantasy
leagues didn't adopt it as a scored week."""


@lru_cache(maxsize=None)
def _weekly_stats_for_season(season: int) -> pd.DataFrame:
    df: pd.DataFrame = nfl.import_weekly_data([season], downcast=True)
    return df.loc[(df["season_type"] == "REG") & (df["week"] <= FANTASY_REGULAR_SEASON_MAX_WEEK)]


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
    return nfl.import_schedules(range(1999, 2030))


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


@lru_cache(maxsize=None)
def _seasonal_roster_for_season(season: int) -> pd.DataFrame:
    return nfl.import_seasonal_rosters([season])


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


def _ensure_pbp_cached(seasons: list[int]) -> None:
    """Play-by-play is nfl_data_py's heaviest download by far, and `import_pbp_data`
    re-fetches it from GitHub every call unless told to use its local disk cache --
    this populates that cache once per season instead of on every call."""
    cache_dir = Path(appdirs.user_cache_dir("nfl_data_py", "cooper_dff")) / "pbp"
    missing = [s for s in seasons if not (cache_dir / f"season={s}").is_dir()]
    if missing:
        nfl.cache_pbp(missing)


def load_red_zone_rush_attempts(seasons: list[int]) -> pd.DataFrame:
    """Per player-week red-zone (inside the 20) rush attempts, from nflverse play-by-play."""
    _ensure_pbp_cached(seasons)
    pbp: pd.DataFrame = nfl.import_pbp_data(seasons, downcast=True, cache=True)
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


INJURY_REPORTS_EARLIEST_SEASON = 2009
"""nflverse's injury report data doesn't go back further than this."""


@lru_cache(maxsize=None)
def _injury_reports_for_season(season: int) -> pd.DataFrame:
    df: pd.DataFrame = nfl.import_injuries([season])
    regular_season = df.loc[df["game_type"] == "REG"]
    return regular_season.rename(columns={"gsis_id": "player_id"})[
        ["player_id", "season", "week", "report_status"]
    ]


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
    return nfl.import_draft_picks()


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
    return nfl.import_snap_counts([season])


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
