import pandas as pd

TEAM_OFFENSIVE_ENVIRONMENT_COLUMNS = [
    "team_pass_attempts_per_game",
    "team_points_per_game",
    "team_passing_ypa",
]


def team_weekly_passing(weekly_all_positions: pd.DataFrame) -> pd.DataFrame:
    """[season, week, team, attempts, passing_yards] -- team-level passing totals per
    week, summed across every player who recorded a pass attempt that week (the starter
    plus any trick-play passer), grouped the same way as `features.add_trailing_team_shares`.
    """
    totals = weekly_all_positions.groupby(["season", "week", "recent_team"])[
        ["attempts", "passing_yards"]
    ].sum()
    return totals.reset_index().rename(columns={"recent_team": "team"})


def season_ending_team_offensive_environment(
    team_weekly_passing_df: pd.DataFrame, team_scores: pd.DataFrame, season: int
) -> pd.DataFrame:
    """[team, team_pass_attempts_per_game, team_points_per_game, team_passing_ypa] -- a
    completed season's team-level offensive environment: pass attempts and passing yards
    per attempt from `team_weekly_passing`, points per game from `nflverse.load_team_scores`.
    The prediction-time counterpart to a trailing average, mirroring
    `strength_of_schedule.season_ending_points_allowed`.
    """
    passing = team_weekly_passing_df.loc[team_weekly_passing_df["season"] == season]
    scores = team_scores.loc[team_scores["season"] == season]

    passing_totals = passing.groupby("team")[["attempts", "passing_yards"]].sum()
    passing_games = passing.groupby("team").size()
    points_per_game = scores.groupby("team")["points"].mean()

    result = pd.DataFrame(
        {
            "team_pass_attempts_per_game": passing_totals["attempts"] / passing_games,
            "team_passing_ypa": passing_totals["passing_yards"] / passing_totals["attempts"],
        }
    )
    result["team_points_per_game"] = points_per_game
    return result.reset_index()[
        ["team", *TEAM_OFFENSIVE_ENVIRONMENT_COLUMNS]
    ]


def team_offensive_environment_history(
    weekly_all_positions: pd.DataFrame, team_scores: pd.DataFrame, seasons: list[int]
) -> pd.DataFrame:
    """[season, team, team_pass_attempts_per_game, team_points_per_game,
    team_passing_ypa] for every season in `seasons` except the earliest (no prior
    completed season to draw from). `season` here is the season the feature applies TO,
    same convention as `strength_of_schedule.season_wide_sos_history`: a row for season
    S carries S-1's completed team-level offensive stats.
    """
    ordered = sorted(seasons)
    passing = team_weekly_passing(weekly_all_positions)

    frames = []
    for season in ordered[1:]:
        season_ending = season_ending_team_offensive_environment(passing, team_scores, season=season - 1)
        if season_ending.empty:
            continue
        season_ending.insert(0, "season", season)
        frames.append(season_ending)

    if not frames:
        return pd.DataFrame(columns=["season", "team", *TEAM_OFFENSIVE_ENVIRONMENT_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def team_offensive_environment_by_player(history: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """[season, player_id, team_pass_attempts_per_game, team_points_per_game,
    team_passing_ypa] -- joins a season-indexed, team-level
    `team_offensive_environment_history` onto each player via their roster team THAT
    season, same dedup pattern as `strength_of_schedule.season_wide_sos_by_player`.
    """
    deduped = rosters.drop_duplicates(subset=["player_id", "season"])
    merged = deduped.merge(history, on=["season", "team"], how="left")
    return merged[["season", "player_id", *TEAM_OFFENSIVE_ENVIRONMENT_COLUMNS]]
