def walk_forward_splits(
    seasons: list[int], min_train_seasons: int
) -> list[tuple[int, int]]:
    """Walk-Forward Backtest splits: train through season N, predict N+1, slide forward.

    Each split trains on all seasons up to and including `train_through_season` and
    predicts `target_season = train_through_season + 1`, so no season's data can leak
    backward into training for a later prediction of an earlier season.
    """
    ordered = sorted(seasons)
    train_through_seasons = ordered[min_train_seasons - 1 : -1]

    if not train_through_seasons:
        raise ValueError(
            f"Need at least {min_train_seasons + 1} seasons to produce one "
            f"Walk-Forward Backtest split; got {len(ordered)}."
        )

    return [(season, season + 1) for season in train_through_seasons]
