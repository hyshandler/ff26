import pandas as pd

from ff_model.feature_cache import cache_key, dataframe_fingerprint, disk_cached_frame
from ff_model.features import season_ending_shares
from ff_model.scoring import PPR

TARGET_SHARE_STAT_COLUMNS = {"target_share": "targets"}


def vacated_target_share_history(
    weekly_all_positions: pd.DataFrame, rosters: pd.DataFrame, seasons: list[int]
) -> pd.DataFrame:
    """One row per (season, player_id): `vacated_target_share`, the sum of prior-season
    target share (per `season_ending_shares`, `add_trailing_team_shares`'s completed-season
    counterpart) held by teammates who were on the player's *prior*-season team but are
    absent from the target season's roster entirely.

    Per issue #23 / ADR-0015: departure detection is pure roster departure only, mirroring
    ADR-0004's judgment-free standard for Depth-Chart Competition's arrival side -- a
    teammate still on *any* team's roster (even injured) doesn't count as departed, and no
    injury/IR/return-status data is consulted. Target share is computed across all
    positions' targets (like `add_trailing_team_shares`'s team totals), since a departing
    RB or TE's share of the team's targets frees up the same pool a WR redistributes into.

    `season` here is the season the feature applies TO, same convention as
    `prior_season_totals`: a row for season S sums shares vacated by players who left
    between S-1 and S. The earliest season in `seasons` has no prior season, so it's
    absent from the result.
    """
    ordered = sorted(seasons)
    deduped = rosters.drop_duplicates(subset=["player_id", "season"])

    frames = []
    for prior_season, season in zip(ordered, ordered[1:]):
        prior_roster = deduped.loc[deduped["season"] == prior_season, ["player_id", "team"]]
        current_ids = set(deduped.loc[deduped["season"] == season, "player_id"])

        departed = prior_roster.loc[~prior_roster["player_id"].isin(current_ids)]
        if departed.empty:
            vacated_by_team = pd.Series(dtype=float)
        else:
            prior_shares = season_ending_shares(
                weekly_all_positions, TARGET_SHARE_STAT_COLUMNS, season=prior_season
            )
            departed = departed.merge(prior_shares, on="player_id", how="left")
            departed["target_share"] = departed["target_share"].fillna(0.0)
            vacated_by_team = departed.groupby("team")["target_share"].sum()

        current_roster = deduped.loc[deduped["season"] == season, ["player_id", "team"]].copy()
        current_roster["vacated_target_share"] = (
            current_roster["team"].map(vacated_by_team).fillna(0.0)
        )
        current_roster.insert(0, "season", season)
        frames.append(current_roster[["season", "player_id", "vacated_target_share"]])

    if not frames:
        return pd.DataFrame(columns=["season", "player_id", "vacated_target_share"])
    return pd.concat(frames, ignore_index=True)


def prior_season_points_per_target(
    weekly_all_positions: pd.DataFrame, seasons: list[int]
) -> pd.DataFrame:
    """One row per (season, player_id): `prior_season_points_per_target`, the receiving
    player's own immediately-preceding-season `(receiving_yards, receiving_tds,
    receptions)` scored via `PPR`'s point-per-unit weights and divided by that season's
    targets.

    Same deliberate, narrow exception to CONTEXT.md's "never bake the scoring formula
    into training" rule as `prior_season_totals` (see ADR-0015). `season` follows the
    same convention: a row for season S carries S-1's rate. A player with zero prior-
    season targets (or no prior season at all) is absent from the result rather than
    given a divide-by-zero value.

    Disk-cached (see `feature_cache`) keyed on `weekly_all_positions`'s content and `seasons`.
    """
    receiving_stats = ["receiving_yards", "receiving_tds", "receptions", "targets"]
    fingerprint = dataframe_fingerprint(weekly_all_positions[["season", "player_id", *receiving_stats]])
    key = cache_key("prior_season_points_per_target", fingerprint, {}, seasons=sorted(seasons))

    def compute() -> pd.DataFrame:
        totals = weekly_all_positions.groupby(["season", "player_id"])[receiving_stats].sum()
        points = (
            totals["receiving_yards"] * PPR.points_per_unit["receiving_yards"]
            + totals["receiving_tds"] * PPR.points_per_unit["receiving_tds"]
            + totals["receptions"] * PPR.points_per_unit["receptions"]
        )
        result = pd.DataFrame(
            {
                "season": totals.index.get_level_values("season") + 1,
                "player_id": totals.index.get_level_values("player_id"),
                "prior_season_points_per_target": points / totals["targets"],
            }
        ).reset_index(drop=True)
        result = result.loc[totals["targets"].to_numpy() > 0]
        return result.loc[result["season"].isin(seasons)].reset_index(drop=True)

    return disk_cached_frame(key, compute)
