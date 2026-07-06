"""Generates notebooks/v2_deliverable.ipynb from source cells.

Run this to regenerate the notebook after `output/v2_projections.csv` or
`output/backtest_report_v2.json` change (see `scripts/build_v2_deliverable.py`);
it's not itself the deliverable, `v2_deliverable.ipynb` (after execution) is.
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
# v2 Fantasy Football Draft Projection Model — Deliverable

Per PRD #11: 2026 projections for all four positions, trained through the completed 2025 season,
led by the corrected v2 scoreboards. Per ADR-0014/issue #19, **Disagreement Edge** is the headline
win-condition metric — not whether the model agrees with ADP overall, but whether the model's
*disagreements* with ADP are trustworthy. Matched-population Spearman and tier accuracy (ADR-0010/
ADR-0011) remain as supporting signals only. The pooled VOR cross-position scoreboard (ADR-0012) was
dropped entirely per issue #15 — only the four per-position scoreboards below.

Rookies are excluded per ADR-0006: every row here is a Veteran (at least one prior season and
>=16 career games before the target season)."""
)

code(
    """\
import json

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)
projections = pd.read_csv("../output/v2_projections.csv", comment="#")
with open("../output/backtest_report_v2.json") as f:
    backtest_report = json.load(f)

projections.shape"""
)

md(
    """\
## Final Projections

All four positions' full Projections for **2026** (trained through the completed 2025 season),
combined into one dataframe — the CSV deliverable itself. Every position now runs the simplest
version of each feature family (no multi-season memory window, no age/experience or
Strength-of-Schedule encoding, no Depth-Chart Competition feature) per issue #16's Disagreement Edge
re-sweep, which found none of the previously-adopted variants distinguishable from noise.
`full_projection_p50` is the season-total point estimate (`fantasy_points_p50 *
games_played_estimate`); `full_projection_p10`/`full_projection_p90` are its boom/bust range."""
)

code(
    """\
projections.groupby("position").size().rename("players_projected")"""
)

code(
    """\
key_columns = [
    "position", "player_name", "target_season", "games_played_estimate",
    "fantasy_points_p10", "fantasy_points_p50", "fantasy_points_p90",
    "full_projection_p10", "full_projection_p50", "full_projection_p90",
]
projections[key_columns].sort_values(
    ["position", "full_projection_p50"], ascending=[True, False]
).groupby("position").head(5)"""
)

md(
    """\
## Boom/Bust Ranges (P10/P50/P90)

Top 12 players per position by `full_projection_p50`, with the P10-P90 range as an error bar."""
)

code(
    """\
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
positions = ["QB", "RB", "WR", "TE"]

for ax, position in zip(axes.flat, positions):
    top = (
        projections.loc[projections["position"] == position]
        .nlargest(12, "full_projection_p50")
        .sort_values("full_projection_p50")
    )
    y = range(len(top))
    lower_err = top["full_projection_p50"] - top["full_projection_p10"]
    upper_err = top["full_projection_p90"] - top["full_projection_p50"]
    ax.errorbar(
        top["full_projection_p50"], y,
        xerr=[lower_err, upper_err],
        fmt="o", capsize=3, color="#2b6cb0", ecolor="#90cdf4",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(top["player_name"])
    ax.set_title(f"{position}: Top 12 by full Projection (P10-P90)")
    ax.set_xlabel("Full Projection (season fantasy points)")

fig.tight_layout()
fig.savefig("../output/v2_boom_bust_ranges.png", dpi=110)
plt.show()"""
)

md(
    """\
## Model vs. live 2026 FFC ADP

Live Fantasy Football Calculator ADP for the 2026 draft class, joined onto the projections by
crosswalked `player_id`. `model_rank`/`adp_rank` are computed per position, over the players FFC's
market has actually drafted (non-null `adp`); `rank_diff` is `model_rank - adp_rank` (negative means
the model likes the player more than the market does). `large_disagreement` flags `|rank_diff| > 12`
— one draft round, the same `LARGE_DISAGREEMENT_THRESHOLD` the Disagreement Edge win-rate below is
scored on (per ADR-0014's note that issue #17's flagged cases and the scored cases should be the
same thing, not two separate ad hoc definitions of "biggest disagreement")."""
)

code(
    """\
matched = projections.dropna(subset=["adp"])
adp_columns = [
    "position", "player_name", "full_projection_p50", "adp", "model_rank", "adp_rank", "rank_diff",
]
matched[adp_columns].sort_values(["position", "adp_rank"]).groupby("position").head(10)"""
)

md(
    """\
### Largest disagreements per position

Where the model and the 2026 market disagree by more than one draft round — the players this model
is actually taking a position on, not just re-stating the crowd's board."""
)

code(
    """\
disagreements = matched.loc[matched["large_disagreement"]]
disagreements[adp_columns].reindex(
    disagreements["rank_diff"].abs().sort_values(ascending=False).index
).groupby("position").head(5)"""
)

md(
    """\
## Walk-Forward Backtest Report (2012-2025)

Per position, computed on the Matched Population (ADR-0010: only players FFC's market actually
drafted that season). **Disagreement Edge (ADR-0014) is the headline win condition**: the model's
*disagreements* with ADP are trustworthy when the win-rate on large disagreements (>12 ranks) is
significantly different from the 50% noise null, per the season-block bootstrap CI. Matched-population
Spearman ρ and tier accuracy vs. ADP are reported alongside as supporting signals only — they measure
overall rank agreement, which ADR-0014 explicitly demoted from decision metric because a feature (or a
whole model) can look good on those by hugging ADP's own ranking, without adding any real disagreement
signal."""
)

code(
    """\
def disagreement_verdict(edge: dict) -> str:
    lo, hi = edge["win_rate_ci"]
    if lo > 0.5:
        return "beats noise"
    if hi < 0.5:
        return "loses to noise"
    return "indistinguishable from noise"


rows = []
for position, report in backtest_report.items():
    edge = report["disagreement_edge"]
    rows.append(
        {
            "position": position,
            "n_matched": report["n_matched"],
            "correlation": edge["correlation"],
            "correlation_ci": edge["correlation_ci"],
            "win_rate": edge["win_rate"],
            "win_rate_ci": edge["win_rate_ci"],
            "n_large_disagreements": edge["n_large_disagreements"],
            "verdict": disagreement_verdict(edge),
        }
    )

disagreement_df = pd.DataFrame(rows).set_index("position")
disagreement_df"""
)

md(
    """\
### Supporting metrics: matched-population Spearman and tier accuracy vs. ADP

Not the win condition (see above) — tracked here for overall model-quality context only."""
)

code(
    """\
supporting_rows = []
for position, report in backtest_report.items():
    tiers = report["tier_accuracy"]
    supporting_rows.append(
        {
            "position": position,
            "model_spearman": report["matched_population"]["model"],
            "adp_spearman": report["matched_population"]["adp"],
            "model_tier_accuracy": tiers["model"]["accuracy"],
            "adp_tier_accuracy": tiers["adp"]["accuracy"],
            "tier_verdict": tiers["verdict"],
        }
    )

pd.DataFrame(supporting_rows).set_index("position")"""
)

md(
    """\
### Reading the Disagreement Edge verdict

QB and TE remain underpowered for this exercise — `n_large_disagreements` is small enough (see
`docs/research/feature-family-re-sweep-2026-07.md`'s Caveats) that a wide win-rate CI is expected,
not a sign the metric is broken. Where the CI excludes 50%, that's the model's disagreements with ADP
carrying real, not noise, information. Where it doesn't, that position's model isn't yet earning its
disagreements with the market — a candidate for v2.1 feature work, not a reason to withhold that
position's projections (v1's precedent: useful, credible per-position signal, not a finished
product)."""
)

nb["cells"] = cells
with open("v2_deliverable.ipynb", "w") as f:
    nbf.write(nb, f)

print("Wrote v2_deliverable.ipynb")
