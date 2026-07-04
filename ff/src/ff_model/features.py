from typing import Literal

import pandas as pd

MultiSeasonWindow = Literal["none", "career", "last_n", "recency_weighted"]


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

    Known gap: a player traded mid-season has their whole-season stat total divided
    by only ONE team's season total (whichever team `drop_duplicates` happens to keep),
    not a per-team-weighted denominator -- unlike `add_trailing_team_shares`, which
    joins by `(season, week, team)` and doesn't have this issue. Left as-is: affects a
    small number of in-season-traded skill players per year, judged low-impact.
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


def _season_totals(weekly: pd.DataFrame, stat_columns: dict[str, str]) -> pd.DataFrame:
    """One row per (season, player_id): summed raw stat totals and games played that season."""
    raw_columns = list(stat_columns.values())
    totals = weekly.groupby(["season", "player_id"])[raw_columns].sum()
    totals["games"] = weekly.groupby(["season", "player_id"]).size()
    return totals.reset_index()


def multi_season_career_averages(
    weekly: pd.DataFrame, stat_columns: dict[str, str], seasons: list[int]
) -> pd.DataFrame:
    """[season, player_id, feature...] per-game average of each stat across every season in
    `seasons` strictly before that row's season -- full career to date, games-weighted (unlike
    `add_trailing_player_averages`, which resets at each season boundary). Shaped like
    `depth_chart.depth_chart_competition_history`: usable both as a per-season training merge
    table and, filtered to `target_season`, as the prediction-time feature directly, since
    `target_season`'s row uses only its strictly-prior seasons and never needs `target_season`'s
    own (not-yet-played) weekly data. The earliest season in `seasons` has no prior season, so
    it's absent from the result.
    """
    ordered = sorted(seasons)
    raw_columns = list(stat_columns.values())
    season_totals = _season_totals(weekly, stat_columns)

    frames = []
    for season in ordered[1:]:
        prior = season_totals.loc[season_totals["season"] < season]
        if prior.empty:
            continue
        totals = prior.groupby("player_id")[raw_columns + ["games"]].sum()
        averages = totals[raw_columns].div(totals["games"], axis=0)
        averages.columns = list(stat_columns.keys())
        averages = averages.reset_index()
        averages.insert(0, "season", season)
        frames.append(averages)

    if not frames:
        return pd.DataFrame(columns=["season", "player_id", *stat_columns.keys()])
    return pd.concat(frames, ignore_index=True)


def multi_season_last_n_averages(
    weekly: pd.DataFrame, stat_columns: dict[str, str], seasons: list[int], n_seasons: int = 3
) -> pd.DataFrame:
    """Like `multi_season_career_averages`, but averaged over only the most recent `n_seasons`
    prior seasons (by season number, not game count) instead of full career."""
    ordered = sorted(seasons)
    raw_columns = list(stat_columns.values())
    season_totals = _season_totals(weekly, stat_columns)

    frames = []
    for season in ordered[1:]:
        window_seasons = [s for s in ordered if s < season][-n_seasons:]
        window = season_totals.loc[season_totals["season"].isin(window_seasons)]
        if window.empty:
            continue
        totals = window.groupby("player_id")[raw_columns + ["games"]].sum()
        averages = totals[raw_columns].div(totals["games"], axis=0)
        averages.columns = list(stat_columns.keys())
        averages = averages.reset_index()
        averages.insert(0, "season", season)
        frames.append(averages)

    if not frames:
        return pd.DataFrame(columns=["season", "player_id", *stat_columns.keys()])
    return pd.concat(frames, ignore_index=True)


def multi_season_recency_weighted_averages(
    weekly: pd.DataFrame, stat_columns: dict[str, str], seasons: list[int], decay: float = 0.5
) -> pd.DataFrame:
    """Like `multi_season_career_averages`, but each prior season's per-game average is weighted
    by `decay ** (seasons_ago - 1)` before averaging, so more recent seasons count for more.
    """
    ordered = sorted(seasons)
    feature_names = list(stat_columns.keys())
    raw_columns = list(stat_columns.values())
    season_totals = _season_totals(weekly, stat_columns)

    per_season_avg = season_totals[raw_columns].div(season_totals["games"], axis=0)
    per_season_avg.columns = feature_names
    per_season_avg["season"] = season_totals["season"]
    per_season_avg["player_id"] = season_totals["player_id"]

    frames = []
    for season in ordered[1:]:
        prior = per_season_avg.loc[per_season_avg["season"] < season].copy()
        if prior.empty:
            continue
        prior["weight"] = decay ** (season - prior["season"] - 1)
        weighted_features = prior[feature_names].mul(prior["weight"], axis=0)
        weighted_features["player_id"] = prior["player_id"]
        weighted_features["weight"] = prior["weight"]
        totals = weighted_features.groupby("player_id").sum()
        averages = totals[feature_names].div(totals["weight"], axis=0).reset_index()
        averages.insert(0, "season", season)
        frames.append(averages)

    if not frames:
        return pd.DataFrame(columns=["season", "player_id", *feature_names])
    return pd.concat(frames, ignore_index=True)


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
