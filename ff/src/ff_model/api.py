"""FastAPI app serving player stats and ADP projections-vs-actuals: a JSON
API plus a single-page HTML/Chart.js UI, both backed by the precomputed
`output/backtest_{POSITION}.csv` files (see `ff_model.player_report`).

Run locally with:

    uv run uvicorn ff_model.api:app --reload --port 8001

or simply:

    uv run python -m ff_model.api

then open http://127.0.0.1:8001/
"""

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from ff_model.player_report import (
    load_combined_backtest,
    player_report,
    top_players_report,
    with_predicted_adp_rank,
)

OUTPUT_DIR = Path(os.environ.get("FF_OUTPUT_DIR", "output"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="ff26 Player Report")


@lru_cache(maxsize=1)
def _combined_backtest() -> pd.DataFrame:
    return with_predicted_adp_rank(load_combined_backtest(OUTPUT_DIR))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/players")
def list_players(q: str = "") -> list[str]:
    """Player names available in the backtest output, optionally filtered by
    a case-insensitive substring `q` (for search-box autocomplete)."""
    names = _combined_backtest()["player_name"].dropna().unique()
    if q:
        names = [name for name in names if q.lower() in name.lower()]
    return sorted(names)[:25]


@app.get("/api/players/{player_name}")
def get_player_report(player_name: str) -> list[dict[str, object]]:
    try:
        report = player_report(_combined_backtest(), player_name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return report.where(pd.notna(report), None).to_dict(orient="records")


@app.get("/api/seasons")
def list_seasons() -> list[int]:
    """Target seasons available in the backtest output, for populating a year
    picker."""
    return sorted(int(season) for season in _combined_backtest()["target_season"].dropna().unique())


@app.get("/api/seasons/{season}/top")
def get_top_players(season: int, position: str | None = None, limit: int = 50) -> list[dict[str, object]]:
    try:
        report = top_players_report(_combined_backtest(), season, position=position, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return report.where(pd.notna(report), None).to_dict(orient="records")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("ff_model.api:app", host="127.0.0.1", port=8001, reload=True)