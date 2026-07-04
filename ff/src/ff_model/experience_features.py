from typing import Literal

import pandas as pd

from ff_model.age_buckets import age_bucket

ExperienceFeature = Literal["none", "age", "years_in_league", "career_games", "career_stage_bucket"]

EXPERIENCE_FEATURE_COLUMN = {
    "age": "age",
    "years_in_league": "years_in_league",
    "career_games": "career_games",
    "career_stage_bucket": "career_stage_bucket",
}
"""experience_feature variant -> the single feature column its history table produces."""


def age_history(rosters: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """[season, player_id, age] -- age is known entering a season (roster metadata,
    not a stat outcome), so unlike the multi-season memory features it needs no
    strictly-prior-season shift to stay leakage-safe.

    A player traded mid-season can appear once per team for the same (player_id,
    season) in nflverse's seasonal rosters (age doesn't differ by team), same dedup
    `games_played.py`/`depth_chart.py` already need.
    """
    deduped = rosters.drop_duplicates(subset=["player_id", "season"])
    return deduped.loc[deduped["season"].isin(seasons), ["season", "player_id", "age"]]


def years_in_league_history(rosters: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """[season, player_id, years_in_league] = season - rookie_year. Also known entering
    the season, no leakage shift needed."""
    deduped = rosters.drop_duplicates(subset=["player_id", "season"])
    result = deduped.loc[
        deduped["season"].isin(seasons), ["season", "player_id", "rookie_year"]
    ].copy()
    result["years_in_league"] = result["season"] - result["rookie_year"]
    return result[["season", "player_id", "years_in_league"]]


def career_stage_bucket_history(rosters: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """[season, player_id, career_stage_bucket] -- the same age-bucket cohorts as the
    Games-Played Estimate heuristic (`age_buckets.age_bucket`), exposed here as a real
    trained-model feature instead of only a heuristic grouping key."""
    deduped = rosters.drop_duplicates(subset=["player_id", "season"])
    result = deduped.loc[deduped["season"].isin(seasons), ["season", "player_id", "age"]].copy()
    result["career_stage_bucket"] = age_bucket(result["age"])
    return result[["season", "player_id", "career_stage_bucket"]]


def career_games_history(weekly: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    """[season, player_id, career_games] -- cumulative games played in seasons STRICTLY
    BEFORE this row's season. Unlike age/years-in-league, this comes from actual box-score
    outcomes, so it needs the same strictly-prior-season shift as
    `features.multi_season_career_averages` to stay leakage-safe (this season's own games
    played is an outcome of the season being predicted, not known in advance).
    """
    ordered = sorted(seasons)
    season_games = (
        weekly.groupby(["season", "player_id"]).size().rename("games").reset_index()
    )

    frames = []
    for season in ordered[1:]:
        prior = season_games.loc[season_games["season"] < season]
        if prior.empty:
            continue
        totals = prior.groupby("player_id")["games"].sum().rename("career_games").reset_index()
        totals.insert(0, "season", season)
        frames.append(totals)

    if not frames:
        return pd.DataFrame(columns=["season", "player_id", "career_games"])
    return pd.concat(frames, ignore_index=True)
