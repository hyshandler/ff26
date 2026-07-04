from typing import Literal

import pandas as pd

from ff_model.scoring import ScoringFormula

SosFeature = Literal["none", "season_wide", "actual_games"]

SOS_FEATURE_COLUMN = {"season_wide": "season_wide_sos", "actual_games": "trailing_sos_faced"}
"""sos_feature variant -> the single feature column its history/pipeline step produces."""


def weekly_points_allowed(weekly: pd.DataFrame, formula: ScoringFormula) -> pd.DataFrame:
    """[season, week, team, position, points_allowed] -- fantasy points scored against
    each team's defense, per position, per week. `team` here is the DEFENSE (the
    `opponent_team` of whichever offensive players' stats are summed), not the offense.
    """
    points = pd.Series(0.0, index=weekly.index)
    for stat, per_unit in formula.points_per_unit.items():
        if stat in weekly.columns:
            points = points + weekly[stat] * per_unit

    df = weekly[["season", "week", "opponent_team", "position"]].copy()
    df["points"] = points
    result = (
        df.groupby(["season", "week", "opponent_team", "position"])["points"].sum().reset_index()
    )
    return result.rename(columns={"opponent_team": "team", "points": "points_allowed"})


def trailing_points_allowed(points_allowed: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe trailing average of `points_allowed`, per (season, team, position),
    blind to the current and later weeks -- same pattern as
    `features.add_trailing_player_averages`, applied to a defense instead of a player.
    """
    df = points_allowed.sort_values(["season", "week"]).reset_index(drop=True)
    df["trailing_points_allowed"] = df.groupby(["season", "team", "position"])[
        "points_allowed"
    ].transform(lambda s: s.shift(1).expanding().mean())
    return df[["season", "week", "team", "position", "trailing_points_allowed"]]


def add_actual_game_sos(weekly: pd.DataFrame, trailing_points_allowed_df: pd.DataFrame) -> pd.DataFrame:
    """Joins each player-week's own opponent's trailing points-allowed-to-their-position
    value onto `weekly` -- 'how tough was this specific week's matchup.' Feeding this
    column into `features.add_trailing_player_averages` gives a per-player trailing SOS
    average automatically aligned to games the player actually played, unlike a
    season-wide schedule average that doesn't know which games got missed.
    """
    return weekly.merge(
        trailing_points_allowed_df.rename(columns={"team": "opponent_team"}),
        on=["season", "week", "opponent_team", "position"],
        how="left",
    )


def season_ending_points_allowed(points_allowed: pd.DataFrame, season: int) -> pd.DataFrame:
    """[team, position, points_allowed_per_game] -- a completed season's average points
    allowed per game to each position by each team; the prediction-time counterpart to
    `trailing_points_allowed` (mirrors `features.season_ending_averages`)."""
    season_df = points_allowed.loc[points_allowed["season"] == season]
    result = season_df.groupby(["team", "position"])["points_allowed"].mean().reset_index()
    return result.rename(columns={"points_allowed": "points_allowed_per_game"})


def season_wide_sos(
    schedule: pd.DataFrame,
    season_ending_points_allowed_df: pd.DataFrame,
    position: str,
    season: int,
) -> pd.DataFrame:
    """[team, season_wide_sos] -- average points-allowed-per-game (to `position`, from
    the most recently completed season) across every opponent on `team`'s full `season`
    schedule. Knowable before `season` starts, since the schedule itself is public --
    unlike `add_actual_game_sos`, this doesn't require knowing which games get played.
    """
    season_schedule = schedule.loc[schedule["season"] == season]
    opponent_strength = season_ending_points_allowed_df.loc[
        season_ending_points_allowed_df["position"] == position
    ].set_index("team")["points_allowed_per_game"]

    merged = season_schedule.copy()
    merged["opponent_strength"] = merged["opponent_team"].map(opponent_strength)
    result = merged.groupby("team")["opponent_strength"].mean().reset_index()
    return result.rename(columns={"opponent_strength": "season_wide_sos"})


def season_wide_sos_by_player(sos_by_team: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """[season, player_id, season_wide_sos] -- joins a season-indexed, team-level
    `season_wide_sos` (e.g. from `season_wide_sos_history`) onto each player via their
    roster team THAT season. A player traded mid-season can appear once per team for
    the same (player_id, season) in nflverse's seasonal rosters; same dedup
    `games_played.py`/`experience_features.py` already need."""
    deduped = rosters.drop_duplicates(subset=["player_id", "season"])
    merged = deduped.merge(sos_by_team, on=["season", "team"], how="left")
    return merged[["season", "player_id", "season_wide_sos"]]


def season_wide_sos_history(
    schedule: pd.DataFrame, points_allowed: pd.DataFrame, position: str, seasons: list[int]
) -> pd.DataFrame:
    """[season, team, season_wide_sos] for every season in `seasons` except the earliest
    (no prior completed season to estimate opponent strength from). Each `season`'s
    value combines that season's public schedule with opponent strength measured from
    `season - 1`'s completed points-allowed data -- the most recent real defensive
    performance available before `season` starts.
    """
    ordered = sorted(seasons)
    frames = []
    for season in ordered[1:]:
        season_ending = season_ending_points_allowed(points_allowed, season=season - 1)
        if season_ending.empty:
            continue
        sos = season_wide_sos(schedule, season_ending, position=position, season=season)
        sos.insert(0, "season", season)
        frames.append(sos)

    if not frames:
        return pd.DataFrame(columns=["season", "team", "season_wide_sos"])
    return pd.concat(frames, ignore_index=True)
