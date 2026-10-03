"""Showcase the outlook: a model card and the standard charts, from `run()`'s output.

    out = outlook.run(...)
    print(report.model_card(out))
    report.dashboard(out).savefig("outlook.png")

Needs matplotlib (not part of the model's own dependencies). Notebook 08 draws these after
every nightly run and logs them to MLflow; tutorials/05 walks through each one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.constants import HOLIDAY_LABELS
from src.outlook.specs import DOWN, UP

COMMUNITY_AREAS_GEOJSON = Path(__file__).resolve().parents[2] / "webapp" / "static" / "data" / "community_areas.geojson"

# The reference palette: two categorical slots, a gray for "normal", and a blue <-> red diverging
# pair with a neutral midpoint ("below normal" <-> "above normal").
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
RED, MID = "#e34948", "#f0efec"
INK, INK_2, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def _axes(ax=None, size=(8, 4)):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=size, facecolor=SURFACE)
    _style(ax)
    return ax


def _pct(x: float, digits: int = 0) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.{digits}f}%"


# ---------------------------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------------------------

def model_card(out: dict[str, pd.DataFrame]) -> str:
    """A Markdown summary of one run: the live week, what's in the models, and how they scored."""
    s = out["summary"].iloc[0]
    live = out["city_weeks"][out["city_weeks"]["is_live"]].iloc[0]
    calls = out["area_weeks"][out["area_weeks"]["is_live"]]["call"].value_counts()
    lb = out["leaderboard"]
    lines = [
        f"## Next-week outlook: week of {s['live_cutoff']}",
        "",
        f"Crime data through {s['data_through']}. Citywide forecast **{_pct(live['pct_vs_normal'], 1)} vs. normal** "
        f"(momentum {_pct(live['pct_momentum'], 1)}, weather {_pct(live['pct_weather'], 1)}, "
        f"calendar {_pct(live['pct_calendar'], 1)}). "
        f"Calls: {calls.get('up', 0)} up, {calls.get('down', 0)} down, {calls.get('unclear', 0)} unclear.",
        "",
        f"### Backtest: {s['test_weeks']} weeks, {s['test_first']} to {s['test_last']}",
        "",
        f"- Citywide: deviance skill {_pct(s['city_skill'], 1)} over the seasonal bar; mean weekly error "
        f"{s['city_error_bar']:.1%} -> {s['city_error_model']:.1%}.",
        f"- Areas: calls on {s['share_called']:.0%} of {s['scored_area_weeks']:,} area-weeks, right "
        f"{s['hit_rate']:.0%} of the time (95% CI {s['hit_lo']:.0%}-{s['hit_hi']:.0%}) against a "
        f"{s['base_rate']:.0%} base rate; Brier skill {_pct(s['brier_skill'], 1)} over climatology.",
        "",
        "### Leaderboard",
        "",
        "| Level | Model | Features | Skill | Brier skill | Called | Right | AUC |",
        "|---|---|---|---|---|---|---|---|",
    ]
    fmt = lambda v, f: "" if v is None or (isinstance(v, float) and math.isnan(v)) else f.format(v)
    for r in lb.itertuples():
        name = f"**{r.model}** (champion)" if r.champion else r.model
        lines.append(f"| {r.level} | {name} | {r.features} | {fmt(r.skill, '{:+.1%}')} | "
                     f"{fmt(getattr(r, 'brier_skill', None), '{:+.1%}')} | {fmt(getattr(r, 'share_called', None), '{:.0%}')} | "
                     f"{fmt(getattr(r, 'hit_rate', None), '{:.0%}')} | {fmt(getattr(r, 'auc', None), '{:.3f}')} |")
    lines += ["", "Skill is the share of the reference's Poisson deviance removed: the seasonal bar for city "
              "models, each area's normal for area models. Calls are up at P(above normal) >= "
              f"{UP:.2f} and down at <= {DOWN:.2f}."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------------------------

def plot_backtest(city_weeks: pd.DataFrame, weeks: int = 104, ax=None):
    """Citywide weekly crime: what happened, the forecast, and the normal, over the last `weeks`
    backtest weeks plus the live one."""
    ax = _axes(ax, (10, 4))
    d = city_weeks.sort_values("cutoff").tail(weeks + 1)
    x = pd.to_datetime(d["cutoff"])
    ax.plot(x, d["bar_calibrated"], color=GRAY, linewidth=1.5, label="Normal (seasonal bar)")
    ax.plot(x, d["y"], color=BLUE, linewidth=2, label="Actual")
    ax.plot(x, d["forecast"], color=ORANGE, linewidth=2, label="Forecast")
    live = d[d["is_live"]]
    if len(live):
        ax.scatter(pd.to_datetime(live["cutoff"]), live["forecast"], s=64, color=ORANGE, edgecolor=SURFACE,
                   linewidth=2, zorder=5)
        ax.annotate("this week", (pd.to_datetime(live["cutoff"]).iloc[0], live["forecast"].iloc[0]),
                    xytext=(-8, 10), textcoords="offset points", ha="right", color=INK_2, fontsize=9)
    ax.set_title("Citywide crime per week: actual vs. forecast", loc="left", fontsize=11)
    ax.set_ylabel("Crimes per week")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1), ncols=3, labelcolor=INK_2)
    return ax


def plot_leaderboard(leaderboard: pd.DataFrame, axes=None):
    """Deviance skill per model, city and area side by side, champions labeled."""
    import matplotlib.pyplot as plt

    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), facecolor=SURFACE)
    for ax, level, ref in zip(axes, ("city", "area"), ("the seasonal bar", "each area's normal")):
        _style(ax)
        d = leaderboard[leaderboard["level"] == level].iloc[::-1]
        y = np.arange(len(d))
        ax.barh(y, d["skill"] * 100, height=0.5, color=BLUE)
        for yi, (v, champ) in enumerate(zip(d["skill"], d["champion"])):
            ax.text(v * 100, yi, f"  {v:+.1%}" + ("  champion" if champ else ""), va="center", fontsize=9,
                    color=INK if champ else INK_2, fontweight="bold" if champ else "normal")
        ax.set_yticks(y, d["model"])
        ax.set_xlim(min(0, d["skill"].min() * 100) - 1, max(d["skill"].max() * 100 * 1.6, 5))
        ax.set_xlabel(f"Deviance skill over {ref} (%)")
        ax.set_title(f"{level.capitalize()} models", loc="left", fontsize=11)
        ax.grid(axis="y", visible=False)
    return axes


def plot_calibration(calibration: pd.DataFrame, ax=None):
    """Reliability: each probability bin's mean P(above normal) against how often areas came in
    above normal. Dot area follows the number of area-weeks in the bin."""
    ax = _axes(ax, (4.5, 4.5))
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, label="Perfectly calibrated")
    size = 30 + 300 * calibration["n"] / calibration["n"].max()
    ax.scatter(calibration["mean_p"], calibration["observed"], s=size, color=BLUE, edgecolor=SURFACE, linewidth=2,
               zorder=3, label="Backtest bins")
    for edge, word in ((DOWN, "down"), (UP, "up")):
        ax.axvline(edge, color=GRID, linewidth=1, zorder=1)
        ax.text(edge, 0.02, f" {word} calls" if word == "up" else f"{word} calls ", ha="left" if word == "up" else "right",
                fontsize=8, color=INK_2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Forecast P(above normal)")
    ax.set_ylabel("Share that came in above normal")
    ax.set_title("Are the probabilities honest?", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK_2)
    return ax


def plot_drivers(city_weeks: pd.DataFrame, ax=None):
    """What moves the live week's citywide forecast: each feature group's effect, in %."""
    ax = _axes(ax, (5, 2.8))
    live = city_weeks[city_weeks["is_live"]].iloc[0]
    groups = [c for c in city_weeks.columns if c.startswith("pct_") and c != "pct_vs_normal"]
    vals = [live[c] * 100 for c in groups] + [live["pct_vs_normal"] * 100]
    labels = [c.removeprefix("pct_").capitalize() for c in groups] + ["Net vs. normal"]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.5, color=[RED if v > 0 else BLUE for v in vals])
    ax.axvline(0, color=AXIS, linewidth=1)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:+.1f}% " if v >= 0 else f" {v:+.1f}% ", va="center", ha="left" if v >= 0 else "right",
                fontsize=9, color=INK_2)
    ax.set_yticks(y, labels)
    lim = max(abs(v) for v in vals) * 1.5 + 0.5
    ax.set_xlim(-lim, lim)
    ax.set_title(f"This week's drivers (week of {live['cutoff']})", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    return ax


def plot_effects(coefficients: pd.DataFrame, ax=None):
    """The live citywide model's fitted effects, as % change in a week's crime: per °F warmer
    and per inch of rain than normal, and for a week containing each holiday."""
    ax = _axes(ax, (5, 5))
    b = coefficients.set_index("feature")["per_unit"]
    rows = {"1 °F warmer than normal": b.get("t_anom", 0.0) * 5 / 9,
            "1 in more rain than normal": b.get("p_anom_cm", 0.0) * 2.54}
    rows |= {HOLIDAY_LABELS.get(f.removeprefix("hol_"), f).removeprefix("the "): v for f, v in b.items()
             if f.startswith("hol_") and f != "hol_in_recent_7d"}
    d = pd.Series({k: math.expm1(v) * 100 for k, v in rows.items()}).sort_values()
    y = np.arange(len(d))
    ax.barh(y, d.to_numpy(), height=0.6, color=[RED if v > 0 else BLUE for v in d])
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_yticks(y, d.index, fontsize=8)
    ax.set_xlabel("% change in the week's crime")
    ax.set_title("What moves a week's crime", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    return ax


def _diverging(p: float):
    """P(above normal) -> blue (0.2 and below) through the neutral gray (0.5) to red (0.8+)."""
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("outlook", [BLUE, MID, RED])
    return cmap(min(max((p - 0.2) / 0.6, 0.0), 1.0))


def plot_area_map(area_weeks: pd.DataFrame, ax=None, geojson: dict | None = None):
    """The live week by community area: P(above normal), blue below and red above, with the
    up and down calls outlined."""
    from matplotlib.patches import Polygon

    ax = _axes(ax, (5, 6.5))
    g = geojson or json.loads(COMMUNITY_AREAS_GEOJSON.read_text())
    live = area_weeks[area_weeks["is_live"]].set_index("community_area")
    for f in g["features"]:
        props = f["properties"]
        area = int(float(props.get("area_numbe") or props.get("area_num_1")))
        p, c = (live.at[area, "p_above"], live.at[area, "call"]) if area in live.index else (np.nan, None)
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            ax.add_patch(Polygon(np.array(poly[0]), closed=True, facecolor=MID if np.isnan(p) else _diverging(p),
                                 edgecolor=INK if c in ("up", "down") else SURFACE,
                                 linewidth=1.2 if c in ("up", "down") else 0.5))
    ax.autoscale_view()
    ax.set_aspect(1 / math.cos(math.radians(41.84)))
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    ax.grid(False)
    week = live["cutoff"].iloc[0] if len(live) else ""
    ax.set_title(f"P(above normal), week of {week}", loc="left", fontsize=11)
    ax.text(0.0, -0.02, "Blue: likely below normal. Red: likely above. Outlined: a call.",
            transform=ax.transAxes, fontsize=8, color=INK_2, va="top")
    return ax


def dashboard(out: dict[str, pd.DataFrame]):
    """The showcase figure: backtest, leaderboard, calibration, drivers and the live map."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 10), facecolor=SURFACE, layout="constrained")
    grid = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 1.2])
    plot_backtest(out["city_weeks"], ax=fig.add_subplot(grid[0, :2]))
    plot_area_map(out["area_weeks"], ax=fig.add_subplot(grid[:2, 2]))
    plot_leaderboard(out["leaderboard"], axes=[fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])])
    plot_calibration(out["calibration"], ax=fig.add_subplot(grid[2, 0]))
    plot_drivers(out["city_weeks"], ax=fig.add_subplot(grid[2, 1]))
    plot_effects(out["coefficients"], ax=fig.add_subplot(grid[2, 2]))
    s = out["summary"].iloc[0]
    fig.suptitle(f"Next-week crime outlook, week of {s['live_cutoff']} (data through {s['data_through']})",
                 x=0.01, ha="left", fontsize=14, color=INK)
    return fig
