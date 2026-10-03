# Neighborhood Watch

A Chicago crime and 311 early-warning app on Databricks: a Spark pipeline (`notebooks/`), shared
Python (`src/`), a FastAPI + HTMX app (`webapp/`) with a tool-calling agent, and Lakebase Postgres
(`lakebase/`).

Read these before changing anything:
- [docs/architecture.md](docs/architecture.md): how the pieces fit, every table, design decisions.
- [docs/operations.md](docs/operations.md): deployment status (what's live, what's pending), the
  nightly job, upkeep, gotchas and known limitations. Keep its status section current.
- [docs/modeling.md](docs/modeling.md): the forecast, events and lead-lag models.

## Conventions

- Shared code lives in `src/` and is imported as `src.*` from the repo root, in notebooks, the app
  and tests alike. Workspace-specific names belong in `src/config.py` only; values the pipeline
  and app must agree on belong in `src/constants.py`.
- Notebooks can't run locally. Keep their logic in `src/` where it can be tested, and say plainly
  what's untested against the live workspace.
- The deployed app installs only `requirements.txt`: nothing it imports may pull in pandas,
  scipy or scikit-learn (`src/outlook/` is notebook-side).
- Leave `src/db_connect.py`'s connection mechanics alone unless asked; changes there need a live
  smoke test (`tests/smoke_test_agent_tools.py`).
- The agent is a plain `openai`-client tool loop. A new tool is an `@_logged` function in
  `src/agent_tools.py`, a `TOOL_SPECS` entry and a `_DISPATCH` line in `src/agent_chat.py`
  (`tests/test_agent_wiring.py` checks they agree). No agent frameworks.
- Model experiments are data in `src/outlook/specs.py`; the leaderboard scores every registered
  spec on the same folds.
- Anything a person reads (UI, agent answers, narratives) is in imperial units; models keep their
  sources' units and convert in the shaping layer. UI titles are sentence case, and copy reads
  the way a person would say it.
- Comments and docs describe the code as it is: no dated history or "used to".

## Checks

```bash
python -m pytest
```

```bash
NW_FAKE_DATA=1 SESSION_SECRET=local-dev-only uvicorn webapp.main:app --reload
```

`python docs/architecture/build_diagrams.py` regenerates the diagrams after editing it.
