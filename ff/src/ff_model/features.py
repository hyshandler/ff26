import pandas as pd


def add_trailing_team_shares(
    weekly_all_positions: pd.DataFrame,
    stat_columns: dict[str, str],
    team_column: str = "recent_team",
) -> pd.DataFrame:
    """Add leakage-safe trailing team-share feature columns.

    For each `feature_name: raw_stat_column` pair, a player-week's feature value is
    the player's cumulative `raw_stat_column` total for the season, divided by the
    team's cumulative total for the same season — both computed over weeks strictly
    before the current week. A player/team's first week of a season has no prior
    weeks to compute a share from, so the feature is NaN there (left for the model
    to handle, not imputed).
    """
    df = weekly_all_positions.sort_values(["season", "week"]).reset_index(drop=True)

    for feature_name, stat_column in stat_columns.items():
        player_prior_cumulative = df.groupby(["season", "player_id"])[stat_column].transform(
            lambda s: s.cumsum().shift(1)
        )

        team_week_totals = (
            df.groupby(["season", "week", team_column])[stat_column]
            .sum()
            .reset_index()
            .sort_values(["season", team_column, "week"])
        )
        team_week_totals["team_prior_cumulative"] = team_week_totals.groupby(
            ["season", team_column]
        )[stat_column].transform(lambda s: s.cumsum().shift(1))

        df = df.merge(
            team_week_totals[["season", "week", team_column, "team_prior_cumulative"]],
            on=["season", "week", team_column],
            how="left",
        )
        df[feature_name] = player_prior_cumulative / df["team_prior_cumulative"]
        df = df.drop(columns="team_prior_cumulative")

    return df


def season_ending_shares(
    weekly_all_positions: pd.DataFrame,
    stat_columns: dict[str, str],
    season: int,
    team_column: str = "recent_team",
) -> pd.DataFrame:
    """Per-player team-share features using a fully completed season's totals.

    Used to build the features a player carries *into* the next, not-yet-played
    season — there's no leakage risk here since `season` is entirely in the past
    relative to the season being projected.
    """
    season_weekly = weekly_all_positions.loc[weekly_all_positions["season"] == season]
    player_team = season_weekly.drop_duplicates("player_id").set_index("player_id")[team_column]

    result = pd.DataFrame({"player_id": player_team.index, team_column: player_team.values})

    for feature_name, stat_column in stat_columns.items():
        player_totals = season_weekly.groupby("player_id")[stat_column].sum()
        team_totals = season_weekly.groupby(team_column)[stat_column].sum()
        result[feature_name] = result["player_id"].map(player_totals) / result[team_column].map(
            team_totals
        )

    return result.drop(columns=team_column)


def add_trailing_player_averages(
    weekly: pd.DataFrame, stat_columns: dict[str, str]
) -> pd.DataFrame:
    """Add leakage-safe trailing per-player average feature columns.

    For each `feature_name: source_column` pair, a player-week's feature value is
    that player's average `source_column` for the season so far, computed over
    weeks strictly before the current week (NaN in a player's first week of a
    season, same as `add_trailing_team_shares`).
    """
    df = weekly.sort_values(["season", "week"]).reset_index(drop=True)
    for feature_name, source_column in stat_columns.items():
        df[feature_name] = df.groupby(["season", "player_id"])[source_column].transform(
            lambda s: s.shift(1).expanding().mean()
        )
    return df


def season_ending_averages(
    weekly: pd.DataFrame, stat_columns: dict[str, str], season: int
) -> pd.DataFrame:
    """Per-player average feature values using a fully completed season's data.

    The prediction-time counterpart to `add_trailing_player_averages`, for the
    same reason `season_ending_shares` complements `add_trailing_team_shares`.
    """
    season_weekly = weekly.loc[weekly["season"] == season]
    result = pd.DataFrame({"player_id": season_weekly["player_id"].unique()})
    for feature_name, source_column in stat_columns.items():
        averages = season_weekly.groupby("player_id")[source_column].mean()
        result[feature_name] = result["player_id"].map(averages)
    return result
