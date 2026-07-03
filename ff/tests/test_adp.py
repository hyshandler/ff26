import pandas as pd
import pytest

from ff_model.adp import crosswalk_adp_to_player_ids, load_adp


@pytest.mark.network
def test_load_adp_returns_expected_columns_for_a_real_season() -> None:
    result = load_adp(2022)

    assert {"name", "position", "team", "adp"} <= set(result.columns)
    assert len(result) > 0
    assert (result["adp"] > 0).all()


def test_crosswalk_matches_on_normalized_name_and_position() -> None:
    adp = pd.DataFrame(
        [
            {"name": "Aaron Jones Sr.", "position": "RB", "team": "GB", "adp": 14.9},
            {"name": "Unmatched Guy", "position": "RB", "team": "GB", "adp": 200.0},
        ]
    )
    rosters = pd.DataFrame(
        [{"player_id": "aj1", "player_name": "Aaron Jones", "position": "RB", "season": 2022}]
    )

    result = crosswalk_adp_to_player_ids(adp, rosters, season=2022)

    matched = result.set_index("name")
    assert matched.loc["Aaron Jones Sr.", "player_id"] == "aj1"
    assert pd.isna(matched.loc["Unmatched Guy", "player_id"])
