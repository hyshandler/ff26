from dataclasses import dataclass

from ff_model.data_availability import assert_no_participation_sourced_features


@dataclass(frozen=True)
class PositionConfig:
    position: str
    raw_stat_columns: list[str]
    """Which raw stats this position's model predicts."""

    share_stat_columns: dict[str, str]
    """feature_name -> underlying raw_stat_column, for trailing team-share features.

    Named `trailing_*` (or otherwise distinct from `data_availability`'s forbidden
    raw-participation names) wherever the underlying signal is participation-sourced,
    since these are the leakage-safe, already-lagged versions of those signals.
    """

    @property
    def needs_red_zone_data(self) -> bool:
        """Whether this position's share features need the play-by-play red-zone pull.

        Derived from `share_stat_columns` rather than stored separately, so a future
        position that adds a red-zone-sourced share column can't forget to flip a flag.
        """
        return "red_zone_carries" in self.share_stat_columns.values()


RB = PositionConfig(
    position="RB",
    raw_stat_columns=[
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
    ],
    share_stat_columns={"carry_share": "carries", "trailing_red_zone_share": "red_zone_carries"},
)

QB = PositionConfig(
    position="QB",
    raw_stat_columns=[
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
    ],
    # A starter's own dropbacks aren't a "share" of anything; a mobile QB's rushing
    # workload relative to the team's total rush attempts is the closest opportunity
    # signal in the same shape as RB's carry_share.
    share_stat_columns={"rush_attempt_share": "carries"},
)

# WR and TE share an identical feature set today (both are pure pass-catchers in
# this model) — this is a real coincidence of the current feature set, not a
# missed abstraction; expect them to diverge once a TE-specific or WR-specific
# signal (e.g. in-line vs. slot usage) is added.
_RECEIVER_RAW_STAT_COLUMNS = ["targets", "receptions", "receiving_yards", "receiving_tds"]
_RECEIVER_SHARE_STAT_COLUMNS = {
    "trailing_target_share": "targets",
    "trailing_air_yards_share": "receiving_air_yards",
}

WR = PositionConfig(
    position="WR",
    raw_stat_columns=_RECEIVER_RAW_STAT_COLUMNS,
    share_stat_columns=_RECEIVER_SHARE_STAT_COLUMNS,
)

TE = PositionConfig(
    position="TE",
    raw_stat_columns=_RECEIVER_RAW_STAT_COLUMNS,
    share_stat_columns=_RECEIVER_SHARE_STAT_COLUMNS,
)

POSITION_CONFIGS = {config.position: config for config in (RB, QB, WR, TE)}

for _config in POSITION_CONFIGS.values():
    assert_no_participation_sourced_features(_config.raw_stat_columns)
    assert_no_participation_sourced_features(list(_config.share_stat_columns))
