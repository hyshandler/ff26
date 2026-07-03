import nfl_data_py as nfl
import pandas as pd


def load_weekly_stats(seasons: list[int]) -> pd.DataFrame:
    """Raw weekly box-score stats from nflverse, at the game/week grain."""
    df: pd.DataFrame = nfl.import_weekly_data(seasons, downcast=True)
    return df


def load_seasonal_rosters(seasons: list[int]) -> pd.DataFrame:
    """Seasonal rosters from nflverse, carrying each player's rookie_year."""
    df: pd.DataFrame = nfl.import_seasonal_rosters(seasons)
    return df
