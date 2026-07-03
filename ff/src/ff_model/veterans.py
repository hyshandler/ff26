import pandas as pd


def veteran_player_ids(
    rosters: pd.DataFrame,
    weekly: pd.DataFrame,
    season: int,
    min_career_games: int,
) -> set[str]:
    """Player IDs eligible as Veterans for `season`.

    A Veteran is not a Rookie in `season` (season != rookie_year) and has played
    at least `min_career_games` games strictly before `season`, per CONTEXT.md.
    """
    roster_ids = set(rosters.loc[rosters["season"] == season, "player_id"])
    rookie_ids = set(
        rosters.loc[
            (rosters["season"] == season) & (rosters["rookie_year"] == season),
            "player_id",
        ]
    )

    career_games = (
        weekly.loc[weekly["season"] < season].groupby("player_id").size()
    )
    veteran_by_games = set(career_games[career_games >= min_career_games].index)

    return (roster_ids - rookie_ids) & veteran_by_games
