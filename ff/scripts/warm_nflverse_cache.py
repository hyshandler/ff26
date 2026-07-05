"""One-time download of every nflverse dataset the pipeline uses, for every season in
`SEASONS`, into the repo-local disk cache (`assets/nflverse_cache/`, see `ff_model.nflverse`).
Run this once per season range so later pipeline/backtest runs never hit the network.
"""

from ff_model.nflverse import (
    load_draft_picks,
    load_injury_reports,
    load_offense_snap_pct,
    load_red_zone_rush_attempts,
    load_schedules,
    load_seasonal_rosters,
    load_weekly_stats,
    pfr_id_crosswalk,
)

SEASONS = list(range(2016, 2026))
"""Every completed season since 2016 through 2025, the most recent completed season."""


def main() -> None:
    print(f"Warming nflverse cache for seasons {SEASONS[0]}-{SEASONS[-1]}...")

    load_weekly_stats(SEASONS)
    print("weekly stats done")

    load_schedules(SEASONS)
    print("schedules done")

    rosters = load_seasonal_rosters(SEASONS)
    print("seasonal rosters done")

    load_draft_picks(SEASONS)
    print("draft picks done")

    load_injury_reports(SEASONS)
    print("injury reports done")

    load_offense_snap_pct(SEASONS, pfr_id_crosswalk(rosters))
    print("snap counts done")

    load_red_zone_rush_attempts(SEASONS)
    print("play-by-play (red zone rush attempts) done")

    print("Cache warm.")


if __name__ == "__main__":
    main()
