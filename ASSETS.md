# Asset Index

Catalog of components, docs, and research that already exist as of the
`archive/pre-reset-2026-07-16` tag. **Check here before building or re-deriving
anything.** If a component is listed, `git show archive/pre-reset-2026-07-16:<path>`
gets you the full file without searching the whole archived history.

This file is intentionally kept outside `CLAUDE.md` and the ticket tracker so it
survives resets of either.

## Source modules (`ff/src/ff_model/`)

| Module | Covers |
|---|---|
| `adp.py` | ADP Benchmark ingestion (Fantasy Football Calculator) |
| `age_buckets.py` | Age-bucketing for player features |
| `api.py` | FastAPI app serving player stats / ADP projections-vs-actuals |
| `backtest.py`, `run_backtest.py`, `backtest_report.py` | Walk-Forward Backtest engine + reporting |
| `data_availability.py` | Data-publication-timing guardrails for the backtest |
| `depth_chart.py` | Depth-Chart Competition Feature (ADR-0004) |
| `disagreement_edge.py` | Disagreement Edge headline metric (ADR-0014) |
| `evaluation.py` | Model evaluation harness |
| `experience_features.py` | Veteran/experience-derived features |
| `feature_audit.py` | Feature auditing utility |
| `feature_cache.py` | Feature computation caching |
| `features.py` | Core feature assembly |
| `games_played.py` | Games-Played Estimate heuristic (ADR-0005) |
| `naive_baseline.py` | Naive baseline for evaluation (ADR-0007) |
| `nflverse.py` | nflverse data source integration (ADR-0003) |
| `opportunity_vacuum.py` | Opportunity Vacuum WR feature (issue #20, ADR-0015) |
| `per_touch_efficiency.py` | Per-Touch Efficiency WR feature (ADR-0015) |
| `pipeline.py` | End-to-end pipeline orchestration |
| `player_report.py` | Per-player Walk-Forward Backtest report |
| `position_config.py`, `position_model.py` | Per-position model config/training (ADR-0002) |
| `quantile_model.py` | Quantile regression for uncertainty (ADR-0002) |
| `rf_model.py` | Random-forest model variant |
| `scoring.py` | Scoring Formula (raw stats → fantasy points) |
| `strength_of_schedule.py` | SOS feature (killed as noise, issue #16 — kept for reference, not reused) |
| `tabfm_model.py` | TabFM (TabPFN-style) evaluation (ADR-0009 — blocked by 10k-row cap, see memory) |
| `team_offensive_environment.py` | Own-Team Offensive Environment WR feature (ADR-0015) |
| `tiers.py` | Tier derivation from actual season outcomes (ADR-0011) |
| `veterans.py` | Veteran/Rookie split (ADR-0006) |

## Notebooks (`ff/notebooks/`)

| File | Purpose |
|---|---|
| `wr_feature_engineering.ipynb` (+ `build_notebook_wr_feature_engineering.py`) | WR feature family round 1 exploration (ADR-0015) |
| `rf_wr_training.ipynb` (+ `build_notebook_rf_wr.py`) | WR random-forest training |
| `v1_deliverable.ipynb` | v1 model deliverable notebook |
| `v2_deliverable.ipynb` (+ `build_notebook_v2.py`) | v2 model deliverable notebook |

## ADRs (`docs/adr/`)

0001 raw-stat target split from Games-Played · 0002 per-position quantile LightGBM ·
0003 nflverse-only data source · 0004 depth-chart feature ships, team-context deferred ·
0005 Games-Played is a heuristic, not a submodel · 0006 rookies excluded from v1 ·
0007 naive baseline + ADP, not scraped projections · 0008 CSV deliverable, API deferred ·
0009 TabFM evaluated alongside LightGBM (blocked) · 0010 Matched Population is the eval population ·
0011 tier accuracy as supporting metric · 0012 VOR-adjusted pooled scoreboard ·
0013 standalone model first, ADP Blend deferred · 0014 Disagreement Edge is the headline metric ·
0015 WR feature engineering round 1 (four new families)

Full text: `docs/adr/000N-*.md` or `git show archive/pre-reset-2026-07-16:docs/adr/`.

## Research (`docs/research/`)

disagreement-edge-metric · fantasy-football-projection-modeling ·
feature-family-re-sweep-2026-07 · historical-adp-and-projections-availability ·
injuries-games-played-correlation · model-vs-adp-matched-population ·
multi-season-memory-features · wr-feature-family-permutation-importance-2026-07

## Tickets open at archive time

- #1 — PRD: v1 fantasy football draft projection model
- #18 — Scrape FantasyPros 2025 ADP as FFC gap-year fallback (blocked: registration-gated)
- #21 — PRD: WR feature engineering round 1

## Known dead ends (don't re-attempt without new information)

- Strength-of-Schedule (opponent-side) — killed as noise, issue #16
- Multi-season memory features — killed, see `docs/research/multi-season-memory-features.md`
- TabFM/TabPFN — blocked by 10k-row training cap (ADR-0009)
- FFC 2025 ADP — no coverage; FantasyPros scrape fallback is registration-gated (issue #18)
