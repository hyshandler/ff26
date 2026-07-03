import pandas as pd

from ff_model.backtest import walk_forward_splits
from ff_model.pipeline import build_position_projections


def run_backtest(position: str, seasons: list[int], min_train_seasons: int) -> pd.DataFrame:
    """Run the Walk-Forward Backtest for one position and concatenate every split's output."""
    splits = walk_forward_splits(seasons, min_train_seasons)

    split_frames = []
    for train_through_season, target_season in splits:
        result = build_position_projections(position, train_through_season, target_season)
        frame = result.projections.copy()
        frame.insert(1, "train_through_season", train_through_season)
        split_frames.append(frame)

    return pd.concat(split_frames, ignore_index=True)
