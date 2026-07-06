"""Generates notebooks/rf_wr_training.ipynb.

Exploratory notebook: train a single RandomForest quantile model (`ff_model.rf_model`,
`model_backend="random_forest"` in `position_model.py`) on WR's current feature set
(ADR-0015 round 1: Prior-Season Total + Opportunity Vacuum, plus the always-on process/
opportunity + trailing-average features) and inspect its predictions and per-stat feature
importances. Not a backtest or a LightGBM comparison -- just "does this train, and what does
it look like" for the RF backend on WR.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text))


md(
    """\
# RF Training Exploration — WR

Trains a single RandomForest quantile model per WR raw stat (`ff_model.rf_model`, the
`model_backend="random_forest"` option in `position_model.py`) on WR's current feature set,
which per ADR-0015 round 1 now includes Prior-Season Total and Opportunity Vacuum alongside
the always-on process/opportunity and trailing-average features (see `CONTEXT.md`).

This is exploratory only: one train/predict pass through `train_through_season`, not a
Walk-Forward Backtest, and no comparison against the LightGBM baseline (ADR-0002). It's here
to sanity-check the RF backend on WR's feature set and look at per-stat feature importances
(a RandomForest, unlike LightGBM's boosted trees, exposes `feature_importances_` directly)."""
)

code(
    """\
import matplotlib.pyplot as plt
import pandas as pd

from ff_model.depth_chart import depth_chart_competition_history
from ff_model.nflverse import (
    load_draft_picks,
    load_offense_snap_pct,
    load_seasonal_rosters,
    load_weekly_stats,
    pfr_id_crosswalk,
)
from ff_model.opportunity_vacuum import vacated_target_share_history
from ff_model.pipeline import EARLIEST_SEASON, MIN_CAREER_GAMES
from ff_model.position_config import POSITION_CONFIGS
from ff_model.position_model import add_position_features, feature_columns
from ff_model.rf_model import train_rf_quantile_model
from ff_model.veterans import veteran_player_ids

pd.set_option("display.max_columns", None)

TRAIN_THROUGH_SEASON = 2025
config = POSITION_CONFIGS["WR"]
weekly_seasons = list(range(EARLIEST_SEASON, TRAIN_THROUGH_SEASON + 1))"""
)

md(
    """\
## Load WR training data

Same ingest `pipeline.build_position_projections` uses for WR: weekly stats, rosters,
snap %, depth-chart competition history, and (WR-specific, per ADR-0015) the Opportunity
Vacuum history. No red-zone data -- WR's `needs_red_zone_data` is `False` (that's a
rushing-share signal, RB/QB-only)."""
)

code(
    """\
weekly_all_positions = load_weekly_stats(weekly_seasons)
weekly = weekly_all_positions.loc[weekly_all_positions["position"] == "WR"]

rosters_all = load_seasonal_rosters(weekly_seasons)
rosters = rosters_all.loc[rosters_all["position"] == "WR"]
eligible = veteran_player_ids(
    rosters, weekly, season=TRAIN_THROUGH_SEASON, min_career_games=MIN_CAREER_GAMES
)

pfr_id_by_player_id = pfr_id_crosswalk(rosters)
snap_pct = load_offense_snap_pct(weekly_seasons, pfr_id_by_player_id)

draft_picks = load_draft_picks(weekly_seasons)
depth_chart_history = depth_chart_competition_history(rosters, draft_picks, weekly_seasons)

opportunity_vacuum_history = vacated_target_share_history(
    weekly_all_positions, rosters_all, weekly_seasons
)

len(weekly), len(eligible)"""
)

code(
    """\
training = add_position_features(
    config,
    weekly_all_positions,
    red_zone_carries=pd.DataFrame(columns=["season", "week", "player_id", "red_zone_carries"]),
    snap_pct=snap_pct,
    depth_chart_history=depth_chart_history,
    opportunity_vacuum_history=opportunity_vacuum_history,
)
training = training.loc[training["season"] <= TRAIN_THROUGH_SEASON]

columns = feature_columns(config)
X_train = training[columns]
columns"""
)

md(
    """\
## Train one RF quantile model per raw stat

WR's raw stats are `targets`, `receptions`, `receiving_yards`, `receiving_tds`
(`_RECEIVER_RAW_STAT_COLUMNS` in `position_config.py`) -- one model per stat, same as the
LightGBM backend, so P10/P50/P90 projections combine stat-by-stat downstream via the
Scoring Formula rather than being predicted directly as fantasy points."""
)

code(
    """\
from ff_model.rf_model import predict_rf_quantiles

models = {}
predictions = {}
for stat in config.raw_stat_columns:
    model = train_rf_quantile_model(X_train, training[stat])
    models[stat] = model
    predictions[stat] = predict_rf_quantiles(model, X_train)

predictions["receiving_yards"].join(training[["player_id", "season", "week", "receiving_yards"]]).head(10)"""
)

md(
    """\
## Feature importances per stat

`RandomForestRegressor.feature_importances_` (mean decrease in impurity across trees) --
a quick read on which features the RF backend is actually splitting on for each target."""
)

code(
    """\
importances = pd.DataFrame(
    {stat: model.feature_importances_ for stat, model in models.items()}, index=columns
)
importances.sort_values("receiving_yards", ascending=False)"""
)

code(
    """\
importances.plot.barh(figsize=(8, 6), width=0.8)
plt.xlabel("Feature importance (mean decrease in impurity)")
plt.title("WR RF quantile model — feature importances per raw stat")
plt.tight_layout()
plt.show()"""
)

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "name": "ff-model",
    "display_name": "ff-model (.venv)",
    "language": "python",
}
with open("rf_wr_training.ipynb", "w") as f:
    nbf.write(nb, f)

print("Wrote rf_wr_training.ipynb")
