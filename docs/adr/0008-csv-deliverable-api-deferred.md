# CSV/dataframe is the v1 deliverable; API is v1.1

Considered building an API alongside the model from the start, to stay flexible for future visualization and tooling. Decided against bundling it into v1: an API commits to an interface for a model that hasn't been validated yet, and building one now risks locking in a shape that's wrong once real backtesting results are in. v1 ships a CSV/dataframe as the source-of-truth artifact plus a notebook for visualization and the backtesting report; once that schema is stable and trusted, v1.1 wraps it in a thin FastAPI layer — a small wrapper by then, not a redesign.

**Outcome**: the v1.1 API landed (`ff/src/ff_model/api.py`, commit a307d5c) — a FastAPI app plus a single-page Chart.js UI reading the precomputed `output/backtest_{POSITION}.csv` files, i.e. exactly the thin wrapper this ADR anticipated, once the CSV schema was stable. This decision is fulfilled, not reversed.
