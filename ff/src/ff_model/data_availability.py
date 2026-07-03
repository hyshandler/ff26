"""Data-publication-timing guardrails for the Walk-Forward Backtest.

Raw box-score stats (carries, rushing_yards, receptions, ...) are final once a
game is played and carry no publication lag. Participation-sourced fields
(snap counts/percentages, target share, air yards share) are compiled from FTN
charting data, which is published up to ~48 hours after a game, and whose
underlying source changed between the pre- and post-2023 seasons. A naive
baseline built only from raw box-score stats has no exposure to either issue;
this guard keeps that true and gives the LightGBM feature-engineering slice
(which does add participation-sourced fields) a place to make the timing
adjustment explicit instead of assuming a plain season/week join is safe.
"""

PARTICIPATION_SOURCED_FEATURES = frozenset(
    {
        "target_share",
        "air_yards_share",
        "wopr",
        "offense_snaps",
        "offense_pct",
        "snap_pct",
        "red_zone_share",
    }
)


def assert_no_participation_sourced_features(columns: list[str]) -> None:
    """Raise if any participation/FTN-sourced column is present in a feature set."""
    offending = PARTICIPATION_SOURCED_FEATURES & set(columns)
    if offending:
        raise ValueError(
            "Participation-sourced feature(s) present without a publication-timing "
            f"adjustment: {sorted(offending)}"
        )
