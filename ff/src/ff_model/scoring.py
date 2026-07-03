from dataclasses import dataclass

import pandas as pd

QUANTILE_SUFFIXES = ("p10", "p50", "p90")


@dataclass(frozen=True)
class ScoringFormula:
    name: str
    points_per_unit: dict[str, float]
    """raw_stat_column -> fantasy points per 1 unit of that stat."""


PPR = ScoringFormula(
    name="ppr",
    points_per_unit={
        "passing_yards": 0.04,
        "passing_tds": 4,
        "interceptions": -2,
        "rushing_yards": 0.1,
        "rushing_tds": 6,
        "receiving_yards": 0.1,
        "receiving_tds": 6,
        "receptions": 1,
    },
)


def score_projections(projections: pd.DataFrame, formula: ScoringFormula) -> pd.DataFrame:
    """Per CONTEXT.md's Scoring Formula: applies `formula` to the model's per-game raw
    stat quantile columns ({stat}_p10/p50/p90), producing fantasy_points_p10/p50/p90.

    Applied downstream of the model, never baked into training, so the same
    projections serve any league's scoring rules. Only stats both scored by
    `formula` and present in `projections` contribute -- a position missing a
    scored stat (e.g. a WR has no passing_yards) simply doesn't add to its total.
    """
    result = projections.copy()
    for suffix in QUANTILE_SUFFIXES:
        points = pd.Series(0.0, index=projections.index)
        for stat, points_per_unit in formula.points_per_unit.items():
            column = f"{stat}_{suffix}"
            if column in projections.columns:
                points = points + projections[column] * points_per_unit
        result[f"fantasy_points_{suffix}"] = points
    return result
