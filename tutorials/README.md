# Tutorials

Seven notebooks that walk through how Neighborhood Watch is designed, one piece at a time. They call
the project's own code in `src/` and `webapp/`, and run locally with no Databricks workspace.

| # | Notebook | What you'll see |
|---|---|---|
| 01 | [Architecture tour](01_architecture_tour.ipynb) | the layers, the table lineage rebuilt from the code, the code layout, config |
| 02 | [Ingestion and the medallion layers](02_ingestion_and_medallion.ipynb) | the SODA client, why bronze is all strings, watermarks, the stub newest day, silver |
| 03 | [Trends and normals](03_trends_and_normals.ipynb) | the partial-month trap, trailing windows, normal for this time of year, crime per resident |
| 04 | [Leading indicators](04_leading_indicators.ipynb) | does 311 lead crime? two-way demeaning, lagged correlations, the placebo |
| 05 | [Next-week outlook](05_next_week_outlook.ipynb) | the seasonal bar, the reporting lag, the backtest, the leaderboard, testing a new feature |
| 06 | [Events look-ahead](06_events_lookahead.ipynb) | observed vs. expected, Gamma-Poisson pooling, the real upcoming list |
| 07 | [The agent and the app](07_agent_and_app.ipynb) | tools, `@_logged`, the real tool loop with a scripted model, the routes |

## Running them

From the repo root:

```bash
python -m pip install -r tutorials/requirements.txt
```

```bash
jupyter lab tutorials/
```

## Data

`_data.py` loads everything in the shape the pipeline's tables have, from the first source available:
1. the research exports in `offline_data/` (see `offline_ml/README.md`), if they're on this machine;
2. otherwise the public APIs (the Chicago Data Portal and Open-Meteo), aggregated on the server so
   only small tables come back, and cached in `tutorials/.cache/`.

The first API run takes about half an hour (the 311 history and the daily crime counts per area since
2001 come a year at a time); after that it's seconds. Set `NW_TUTORIAL_SOURCE=api` to skip the local exports. Notebook 02 always calls
the live APIs, and notebook 06's real look-ahead needs the research event exports.
