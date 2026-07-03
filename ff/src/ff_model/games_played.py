import pandas as pd

AGE_BUCKET_EDGES = [0, 24, 27, 30, 100]
AGE_BUCKET_LABELS = ["<24", "24-26", "27-29", "30+"]

RECENT_INJURY_WEEKS = (15, 16, 17)
"""The final weeks of a REG season (per FANTASY_REGULAR_SEASON_MAX_WEEK) used as the
'finished the season hurt' proxy, since nflverse has no clean IR/season-ending flag."""


def _age_bucket(age: pd.Series) -> pd.Series:
    return pd.cut(age, bins=AGE_BUCKET_EDGES, labels=AGE_BUCKET_LABELS, right=False)


def _recently_injured_player_ids(injury_reports: pd.DataFrame, season: int) -> set[str]:
    reports = injury_reports.loc[
        (injury_reports["season"] == season)
        & (injury_reports["week"].isin(RECENT_INJURY_WEEKS))
        & (injury_reports["report_status"] == "Out")
    ]
    return set(reports["player_id"])


def estimate_games_played(
    rosters: pd.DataFrame,
    weekly: pd.DataFrame,
    injury_reports: pd.DataFrame,
    train_through_season: int,
    target_season: int,
    player_ids: set[str],
) -> pd.Series:
    """Games-Played Estimate per CONTEXT.md/ADR-0005: an empirical (position, age_bucket)
    base rate, split by a recent-major-injury flag, computed only from seasons through
    `train_through_season`."""
    training_rosters = rosters.loc[rosters["season"] <= train_through_season].copy()
    training_rosters["age_bucket"] = _age_bucket(training_rosters["age"])

    games_played = weekly.groupby(["player_id", "season"]).size().rename("games")
    training_rosters = training_rosters.merge(
        games_played, on=["player_id", "season"], how="left"
    )
    training_rosters["games"] = training_rosters["games"].fillna(0)

    flagged_ids = _recently_injured_player_ids(injury_reports, train_through_season)
    training_rosters["flagged"] = training_rosters["player_id"].isin(flagged_ids)

    base_rate = training_rosters.groupby(["position", "age_bucket", "flagged"], observed=True)[
        "games"
    ].mean()

    target_rosters = rosters.loc[rosters["season"] == target_season].copy()
    target_rosters["age_bucket"] = _age_bucket(target_rosters["age"])
    target_rosters["flagged"] = target_rosters["player_id"].isin(flagged_ids)
    target_rosters = target_rosters.loc[target_rosters["player_id"].isin(player_ids)]

    estimates = target_rosters.set_index("player_id").apply(
        lambda row: base_rate.get((row["position"], row["age_bucket"], row["flagged"])), axis=1
    )
    return estimates.rename("games_played_estimate")
