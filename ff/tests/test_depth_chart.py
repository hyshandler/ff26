import pandas as pd

from ff_model.depth_chart import depth_chart_competition_flag, depth_chart_competition_history

EMPTY_DRAFT_PICKS = pd.DataFrame(columns=["player_id", "season", "round", "position", "team"])


def _rosters(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_flags_returning_players_when_a_veteran_switches_teams_into_their_group() -> None:
    rosters = _rosters(
        [
            {"player_id": "incumbent", "season": 2021, "team": "DAL", "position": "RB", "rookie_year": 2018},
            {"player_id": "incumbent", "season": 2022, "team": "DAL", "position": "RB", "rookie_year": 2018},
            {"player_id": "signee", "season": 2021, "team": "NYG", "position": "RB", "rookie_year": 2017},
            {"player_id": "signee", "season": 2022, "team": "DAL", "position": "RB", "rookie_year": 2017},
        ]
    )

    result = depth_chart_competition_flag(
        rosters, EMPTY_DRAFT_PICKS, train_through_season=2021, target_season=2022
    )

    assert bool(result["incumbent"]) is True


def test_flags_returning_players_when_a_high_round_rookie_is_drafted_at_their_position() -> None:
    rosters = _rosters(
        [
            {"player_id": "incumbent", "season": 2021, "team": "DAL", "position": "RB", "rookie_year": 2018},
            {"player_id": "incumbent", "season": 2022, "team": "DAL", "position": "RB", "rookie_year": 2018},
            {"player_id": "rookie", "season": 2022, "team": "DAL", "position": "RB", "rookie_year": 2022},
        ]
    )
    draft_picks = pd.DataFrame(
        [{"player_id": "rookie", "season": 2022, "round": 1, "position": "RB", "team": "DAL"}]
    )

    result = depth_chart_competition_flag(
        rosters, draft_picks, train_through_season=2021, target_season=2022
    )

    assert bool(result["incumbent"]) is True
    # The rookie arrival itself doesn't get flagged against its own arrival.
    assert bool(result["rookie"]) is False


def test_does_not_flag_for_a_late_round_rookie() -> None:
    rosters = _rosters(
        [
            {"player_id": "incumbent", "season": 2021, "team": "DAL", "position": "RB", "rookie_year": 2018},
            {"player_id": "incumbent", "season": 2022, "team": "DAL", "position": "RB", "rookie_year": 2018},
            {"player_id": "late_pick", "season": 2022, "team": "DAL", "position": "RB", "rookie_year": 2022},
        ]
    )
    draft_picks = pd.DataFrame(
        [{"player_id": "late_pick", "season": 2022, "round": 6, "position": "RB", "team": "DAL"}]
    )

    result = depth_chart_competition_flag(
        rosters, draft_picks, train_through_season=2021, target_season=2022
    )

    assert bool(result["incumbent"]) is False


def test_history_computes_the_flag_for_every_consecutive_season_transition() -> None:
    rosters = _rosters(
        [
            {"player_id": "incumbent", "season": 2020, "team": "DAL", "position": "RB", "rookie_year": 2018},
            {"player_id": "incumbent", "season": 2021, "team": "DAL", "position": "RB", "rookie_year": 2018},
            {"player_id": "incumbent", "season": 2022, "team": "DAL", "position": "RB", "rookie_year": 2018},
            {"player_id": "signee", "season": 2020, "team": "NYG", "position": "RB", "rookie_year": 2017},
            {"player_id": "signee", "season": 2021, "team": "NYG", "position": "RB", "rookie_year": 2017},
            {"player_id": "signee", "season": 2022, "team": "DAL", "position": "RB", "rookie_year": 2017},
        ]
    )

    result = depth_chart_competition_history(
        rosters, EMPTY_DRAFT_PICKS, seasons=[2020, 2021, 2022]
    )

    by_season = result.set_index(["season", "player_id"])["depth_chart_competition"]
    # No prior season to compare 2021 against 2020's roster for `incumbent`: unflagged.
    assert bool(by_season[(2021, "incumbent")]) is False
    # signee switches teams for 2022 -> incumbent (still on DAL) is flagged for 2022.
    assert bool(by_season[(2022, "incumbent")]) is True
