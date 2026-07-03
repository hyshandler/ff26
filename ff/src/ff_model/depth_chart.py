import pandas as pd

HIGH_DRAFT_PICK_MAX_ROUND = 3
"""A drafted rookie counts as new depth-chart competition only within this round."""


def depth_chart_competition_flag(
    rosters: pd.DataFrame,
    draft_picks: pd.DataFrame,
    train_through_season: int,
    target_season: int,
) -> pd.Series:
    """Per ADR-0004: flags every returning player whose (team, position) group gained
    new offseason competition for touches -- a free-agent/trade arrival (a Veteran
    whose team changed from `train_through_season` to `target_season`) or a rookie
    drafted in `target_season` within `HIGH_DRAFT_PICK_MAX_ROUND`. Mechanically
    derived from roster composition; no subjective scheme/fit judgment.
    """
    deduped = rosters.drop_duplicates(subset=["player_id", "season"])
    prior = deduped.loc[deduped["season"] == train_through_season].set_index("player_id")["team"]
    current = deduped.loc[deduped["season"] == target_season].copy()

    is_rookie = current["rookie_year"] == target_season
    prior_team = current["player_id"].map(prior)
    switched_teams = ~is_rookie & (prior_team != current["team"])

    high_picks = draft_picks.loc[
        (draft_picks["season"] == target_season)
        & (draft_picks["round"] <= HIGH_DRAFT_PICK_MAX_ROUND)
    ]
    high_pick_ids = set(high_picks["player_id"])
    drafted_high = is_rookie & current["player_id"].isin(high_pick_ids)

    current["is_new_arrival"] = switched_teams | drafted_high

    new_arrival_groups = set(
        map(tuple, current.loc[current["is_new_arrival"], ["team", "position"]].to_numpy())
    )

    def _flagged(row: pd.Series) -> bool:
        group = (row["team"], row["position"])
        return group in new_arrival_groups and not row["is_new_arrival"]

    flags = current.apply(_flagged, axis=1)
    flags.index = current["player_id"]
    return flags


def depth_chart_competition_history(
    rosters: pd.DataFrame, draft_picks: pd.DataFrame, seasons: list[int]
) -> pd.DataFrame:
    """`depth_chart_competition_flag` for every consecutive season transition in
    `seasons`, long-format as [season, player_id, depth_chart_competition] -- a
    per-player-week training feature needs the flag recomputed for every historical
    season, not just the final train_through_season -> target_season transition.
    The earliest season in `seasons` has no prior season to compare against, so it's
    absent from the result.
    """
    ordered = sorted(seasons)
    frames = []
    for prior_season, season in zip(ordered, ordered[1:]):
        flags = depth_chart_competition_flag(
            rosters, draft_picks, train_through_season=prior_season, target_season=season
        )
        frame = flags.rename("depth_chart_competition").reset_index()
        frame.columns = ["player_id", "depth_chart_competition"]
        frame.insert(0, "season", season)
        frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["season", "player_id", "depth_chart_competition"])
    return pd.concat(frames, ignore_index=True)
