# Model vs. ADP: The Matched-Population Correction

Analysis date: 2026-07-04. Corrects a methodology bug in every prior "model beats ADP"
comparison reported in the v1 backtest report.

## What was wrong

`with_adp_benchmark` (`run_backtest.py`) reset its entire `adp` column (`result["adp"] = pd.NA`)
on every call. The backtest-rebuild scripts call it once per season, chaining the result across
seasons 2020-2024 (`backtest = with_adp_benchmark(backtest, adp, season=target)` in a loop) --
each call silently wiped out every earlier season's joined ADP values. Only the last season
processed (2024) survived, so `n_with_adp` in every backtest report to date was really "how many
2024 players have ADP," not "how many 2020-2024 players have ADP." Fixed by only initializing the
`adp` column if it doesn't already exist (`ff_model/run_backtest.py`), with a regression test
(`test_with_adp_benchmark_preserves_earlier_seasons_when_called_repeatedly`) that chains two calls
and asserts both seasons' values survive.

Separately, and independent of that bug: every "model beats ADP" comparison computed the model's
Spearman ρ over the **full backtest population** (every eligible Veteran, hundreds of players
including irrelevant depth pieces) while computing ADP's ρ over only the **small ADP-matched
subset** (players relevant enough to have real crowd-consensus ADP). These are different
populations, so the comparison was never apples-to-apples.

## Corrected result

With the bug fixed, `n_with_adp` is now the true 2020-2024 count (3-5x larger than before), and
computing the model's ρ on that *same* matched subset (`model_spearman_vs_actual_adp_matched` in
`output/backtest_report.json`) gives a very different picture than the full-population number:

| Position | n_with_adp (corrected) | Model ρ (full population) | Model ρ (ADP-matched) | ADP ρ (matched) |
|---|---|---|---|---|
| QB | 96 | 0.577 | **0.019** | 0.491 |
| RB | 211 | 0.582 | 0.445 | 0.534 |
| WR | 250 | 0.628 | 0.252 | 0.560 |
| TE | 80 | 0.588 | 0.299 | 0.452 |

**ADP beats the model on rank correlation for all four positions** once measured on the same
population. QB is the starkest case: among ADP-relevant QBs, the model's correlation with actual
finish order is statistically indistinguishable from zero.

## Why the full-population number was misleading

The model's full-population ρ is inflated by an easy sub-task: correctly ranking obvious stars far
above irrelevant bench players is trivial signal that any reasonable model gets right, and it
dominates a correlation computed over a population that's mostly lopsided talent gaps. The ADP-
matched subset removes that easy signal — it's good player vs. good player, which is the actual
question a draft decision turns on — and that's where the model currently has little to no edge
over, or is worse than, simply following crowd consensus.

## Conclusion

The "model beats ADP for 3 of 4 positions" framing used earlier this session does not survive an
apples-to-apples check and should be retired. The credible, corrected finding is the opposite: for
the population that matters for a real draft decision, ADP currently outperforms the model on rank
order at every position. This is a more consequential finding than any of the individual feature
experiments run today (multi-season memory, age/experience, Strength of Schedule) and should weigh
more heavily on what gets prioritized next -- closing this gap is arguably a bigger opportunity than
incremental feature additions on top of a model that isn't yet beating the crowd where it counts.
