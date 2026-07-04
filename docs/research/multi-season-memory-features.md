# Multi-Season Memory Features: Backtest Comparison and Decision

Analysis date: 2026-07-03. Triggered by feature-importance inspection of the v1 model, which
surfaced a structural gap: the single-season `trailing_avg_*` features (`features.py`'s
`add_trailing_player_averages` / `season_ending_averages`) reset at every season boundary, so a
Veteran's prediction-time feature reflects only their *most recent completed season* — while the
naive baseline it's benchmarked against (`naive_baseline.py`) averages over the player's **entire
prior career**. Since trailing averages were the dominant feature category by LightGBM gain
importance at every position (18-27% on the single largest feature alone), this narrower window
was a plausible explanation for the model losing to the naive baseline on MAE at QB/RB/TE (see the
v1 deliverable's backtest report).

## What was built

Three multi-season variants, added alongside (not replacing) the existing single-season
`trailing_avg_*` features, all shaped like `depth_chart.depth_chart_competition_history` — a
`[season, player_id, feature...]` table computed once per season from strictly-prior seasons, so
the same table serves both the per-season training merge and the prediction-time lookup
(`features.py`):

- **`multi_season_career_averages`** — full career to date, games-weighted (the direct match for
  the naive baseline's window).
- **`multi_season_last_n_averages`** (default `n_seasons=3`) — most recent N prior seasons only,
  to avoid diluting a player's current role with very old, less-relevant seasons.
- **`multi_season_recency_weighted_averages`** (default `decay=0.5`) — exponentially-weighted
  average across all prior seasons, most recent weighted highest.

## Methodology

Walk-Forward Backtest, seasons 2018-2024 (min 3 training seasons — 4 splits: train-through
2020/2021/2022/2023, predicting 2021/2022/2023/2024 respectively). 2024 is included as a genuine
holdout here for the first time: nflverse's weekly data confirms full 2024 season data (17 weeks)
is available, while 2025 is not yet published (`import_weekly_data([2025])` 404s) — so 2024 is
currently the most recent valid holdout season, and the v1 deliverable's existing "2024 projection"
(built as a forward-looking forecast before 2024 had been played) is actually backtestable now, not
a genuine future forecast.

Each position was run four times — `none` (baseline, single-season only), `career`, `last_n`,
`recency_weighted` — comparing MAE (vs. `actual_fantasy_points`) and Spearman ρ (vs.
`actual_fantasy_points`) per variant. Full numbers: `output/multi_season_sweep.json`.

## Result

| Position | Variant | MAE | Δ vs. baseline | Spearman ρ | Δ vs. baseline |
|---|---|---|---|---|---|
| QB (n=201) | none (baseline) | 95.59 | — | 0.444 | — |
| QB | career | 94.12 | -1.5% | 0.445 | +0.001 |
| QB | **last_n** | **92.20** | **-3.5%** | **0.514** | **+0.070** |
| QB | recency_weighted | 93.03 | -2.7% | 0.476 | +0.032 |
| RB (n=343) | none (baseline) | 62.75 | — | 0.6445 | — |
| RB | career | 62.23 | -0.8% | 0.6548 | +0.010 |
| RB | last_n | 62.76 | +0.0% | 0.6546 | +0.010 |
| RB | **recency_weighted** | **61.52** | **-2.0%** | **0.6551** | **+0.011** |
| WR (n=565) | none (baseline) | 58.36 | — | 0.6555 | — |
| WR | career | 57.72 | -1.1% | 0.6608 | +0.005 |
| WR | last_n | 56.91 | -2.5% | 0.6550 | -0.001 |
| WR | **recency_weighted** | **56.90** | **-2.5%** | **0.6625** | **+0.007** |
| TE (n=271) | none (baseline) | 45.32 | — | 0.6353 | — |
| TE | career | 45.25 | -0.2% | 0.6372 | +0.002 |
| TE | last_n | 44.95 | -0.8% | 0.6369 | +0.002 |
| TE | **recency_weighted** | **44.76** | **-1.1%** | **0.6504** | **+0.015** |

## Decision

Per the same rule already used for the Depth-Chart Competition Feature and the TabFM comparison
("only adopt a change if it clearly wins"): every position improved on **both** MAE and Spearman ρ
with its best-performing multi-season variant, unlike the Depth-Chart Competition Feature's
negligible RB effect. This is a clear, consistent win, not a wash — adopted as each position's new
default:

- **QB → `last_n` (3 seasons)**: biggest rank-correlation gain of any position/variant pair
  (+0.07), and closes most of the remaining MAE gap against the naive baseline (92.20 vs. 91.78 —
  effectively tied, down from a 3.8-point gap).
- **RB, WR, TE → `recency_weighted` (decay 0.5)**: best or tied-best on both metrics for all three.

Wired as `PositionConfig.multi_season_window` (`position_config.py`) rather than hardcoded in a
script, so it's the new default for every caller of `build_position_projections` /
`build_position_model_projections`; passing `multi_season_window="none"` explicitly still
reproduces the original single-season-only baseline for future ablation work.

## Caveats

- QB's n (201) is the smallest of the four positions — the +0.07 rho gain is the largest effect in
  this table and should be treated as encouraging rather than conclusive until it holds up over
  more backtest splits (more seasons become available for holdout each year).
- This backtest only compares multi-season memory variants against each other and the existing
  single-season baseline — it does not re-run the ADP Benchmark or naive-baseline comparison from
  the v1 deliverable notebook, which should be refreshed to reflect these new defaults (see
  `notebooks/v1_deliverable.ipynb`).
