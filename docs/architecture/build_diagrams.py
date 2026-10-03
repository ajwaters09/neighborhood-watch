"""Build the architecture diagrams: one definition, two outputs.

    python docs/architecture/build_diagrams.py

- architecture.drawio: every diagram as a page, editable in diagrams.net (draw.io) and importable
  into Lucidchart (File -> Import -> draw.io).
- <page>.svg: the same diagrams as static images, for the README and docs.

Each diagram is laid out by hand below (boxes, lanes, tables and edges with explicit waypoints),
so both outputs match exactly. Edit the coordinates here and re-run rather than editing the
outputs; edits made in draw.io or Lucidchart won't flow back.

Standard library only.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

OUT = Path(__file__).resolve().parent
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
INK, INK_2, MUTED, EDGE, PAGE = "#1b1b1a", "#55534e", "#898781", "#6b6a65", "#fcfcfb"

# Layer palette: (fill, stroke, lane fill, lane title color).
STYLES = {
    "source": ("#ffffff", "#9a988f", "#f4f3ef", "#55534e"),
    "bronze": ("#fbefe3", "#c4813f", "#fdf8f3", "#8f5420"),
    "silver": ("#eef2f6", "#7d8b9b", "#f7f9fb", "#4c5866"),
    "gold": ("#fcf4d8", "#c29a22", "#fefbf0", "#7d6210"),
    "serve": ("#e6f0fc", "#2a78d6", "#f5f9fe", "#1c5cab"),
    "app": ("#e3f5ee", "#1b9e6f", "#f4fbf8", "#11704d"),
    "user": ("#ffffff", "#55534e", "#ffffff", "#1b1b1a"),
    "analytics": ("#f1eefc", "#6f5bd0", "#f9f8fe", "#4a3aa7"),
    "note": ("#ffffff", "#c3c2b7", "#ffffff", "#55534e"),
    "manual": ("#ffffff", "#898781", "#ffffff", "#55534e"),
}


@dataclass
class Lane:
    x: float
    y: float
    w: float
    h: float
    title: str
    style: str


@dataclass
class Box:
    id: str
    x: float
    y: float
    w: float
    h: float
    title: str
    lines: tuple[str, ...] = ()
    style: str = "source"
    dashed: bool = False


@dataclass
class Table:
    id: str
    x: float
    y: float
    w: float
    title: str
    columns: list[tuple[str, str]]     # (name, type); a name starting "PK " or "FK " gets a key mark
    style: str = "serve"
    row_h: float = 17
    head_h: float = 28

    @property
    def h(self) -> float:
        return self.head_h + self.row_h * len(self.columns) + 6


@dataclass
class Edge:
    points: list[tuple[float, float]]
    label: str | None = None
    label_at: tuple[float, float] | None = None   # default: the middle of the longest segment
    source: str | None = None
    target: str | None = None
    start_arrow: bool = False
    end_arrow: bool = True
    dashed: bool = False


@dataclass
class Text:
    x: float
    y: float
    text: str
    size: float = 12
    bold: bool = False
    color: str = INK_2
    anchor: str = "start"


@dataclass
class Page:
    name: str
    file: str
    width: float
    height: float
    title: str
    subtitle: str
    items: list = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Page 1: the system
# ---------------------------------------------------------------------------------------------

def system() -> Page:
    p = Page("System", "system", 1670, 960, "Neighborhood Watch: system architecture",
             "Chicago crime + 311 early warning on Databricks. Notebook numbers in parentheses; arrows show data flow.")
    lanes = [(20, 200, "SOURCES", "source"), (250, 220, "INGEST: BRONZE", "bronze"), (500, 210, "CLEAN: SILVER", "silver"),
             (740, 240, "GOLD AND MODELS", "gold"), (1010, 220, "SERVING", "serve"), (1260, 220, "DATABRICKS APP", "app"),
             (1510, 140, "USERS", "source")]
    for x, w, title, style in lanes:
        p.items.append(Lane(x, 70, w, 680, title, style))
    p.items.append(Lane(250, 780, 980, 140, "ANALYTICS LOOP", "analytics"))

    B = lambda id, lane_x, lane_w, y, h, title, *lines, style, dashed=False: p.items.append(
        Box(id, lane_x + 15, y, lane_w - 30, h, title, lines, style, dashed))
    # sources
    B("wiki", 20, 200, 110, 56, "Wikipedia", "MediaWiki API · 77 articles", style="source")
    B("portal", 20, 200, 228, 84, "Chicago Data Portal", "SODA API", "crimes · 311 requests", "street + park permits", style="source")
    B("meteo", 20, 200, 334, 62, "Open-Meteo", "observed + forecast weather", style="source")
    B("music", 20, 200, 418, 62, "setlist.fm · Ticketmaster", "club shows, upcoming events", style="source")
    # bronze
    B("b_wiki", 250, 220, 110, 56, "bronze_wiki_pages (02d)", "only changed revisions", style="bronze")
    B("b_vol", 250, 220, 200, 56, "UC Volume: landing", "bulk crime CSV · seed files", style="bronze")
    B("b_main", 250, 220, 280, 116, "Bronze Delta tables", "bronze_crimes · bronze_311", "weather · permits · setlists",
      "all-string, watermarked", "(01, 01b, 01c, 02, 02b, 02c)", style="bronze")
    # silver
    B("s_wiki", 500, 210, 110, 56, "silver_wiki_chunks (03c)", "~940 passages", style="silver")
    B("s_main", 500, 210, 230, 62, "silver_crimes · silver_311", "typed · deduplicated (03)", style="silver")
    B("s_cats", 500, 210, 316, 62, "crime_category_map", "ai_classify, once per pair (03b)", style="silver")
    B("s_events", 500, 210, 402, 62, "silver_events (08b)", "H3 cells, Spark build", style="silver")
    # gold
    B("g_outlook", 740, 240, 190, 62, "outlook_* (08)", "Poisson GLM + NB calls · MLflow", style="gold")
    B("g_events", 740, 240, 268, 62, "events_upcoming (08b)", "Gamma-Poisson look-ahead", style="gold")
    B("g_lead", 740, 240, 346, 62, "indicator_lead_lag (05)", "311 → crime lead-lag", style="gold")
    B("g_narr", 740, 240, 424, 62, "area_narratives (09)", "ai_query, nightly", style="gold")
    B("g_ctx", 740, 240, 502, 62, "area_trend_normals (04)", "+ area_profile: per resident", style="gold")
    B("g_trend", 740, 240, 596, 72, "area_trend_metrics (04)", "+ area_trend_rolling", "monthly · 30/60/90-day windows", style="gold")
    # serving
    B("vs", 1010, 220, 110, 56, "Vector Search index", "hybrid · gte-large embeddings", style="serve")
    B("fm", 1010, 220, 190, 62, "Foundation Model API", "Claude Sonnet: chat, ai_query", style="serve")
    B("wh", 1010, 220, 350, 72, "SQL Warehouse", "Statement Execution API", "small gold tables, cached 1 h", style="serve")
    B("lb", 1010, 220, 548, 120, "Lakebase Postgres", "Synced Tables: *_lb", "users · subscriptions · reports",
      "alert_rules · tool_invocations", "chat_turns", style="serve")
    # app
    B("agent", 1260, 220, 110, 150, "Chat agent", "OpenAI-protocol tool loop", "8 read tools · 2 write tools",
      "no agent framework", "every call logged", style="app")
    B("ui", 1260, 220, 330, 338, "Web app", "FastAPI · HTMX · Alpine", "Leaflet map · Chart.js", "",
      "Explore: map + area panel", "My Areas: subscriptions, alerts", "Insights: forecast drivers,", "events, 311 lead-lag",
      "", "fake-data mode for local work", style="app")
    B("users", 1510, 140, 440, 60, "Residents", "in the browser", style="user")
    # orchestration
    p.items.append(Box("job", 265, 690, 700, 44, "Lakeflow Job, nightly: pulls → silver → gold → models",
                       ("dependency graph on the next page",), "note"))
    # analytics loop
    B("hist", 1010, 220, 828, 62, "lb_<table>_history", "Delta, via Lakebase CDF", style="analytics")
    B("events", 740, 240, 828, 62, "subscription_events (06)", "subscribe · unsubscribe · report", style="analytics")
    B("views", 500, 210, 828, 62, "Analytics views (07)", "tool usage · chat quality", style="analytics")
    B("dash", 250, 220, 828, 62, "Dashboards and evals", "Databricks SQL · MLflow", style="analytics")

    E = lambda *pts, **kw: p.items.append(Edge(list(pts), **kw))
    # sources -> bronze
    E((215, 138), (265, 138), source="wiki", target="b_wiki")
    E((215, 250), (235, 250), (235, 228), (265, 228), source="portal", target="b_vol", label="bulk CSV", label_at=(235, 214))
    E((215, 290), (235, 290), (235, 338), (265, 338), source="portal", target="b_main")
    E((215, 365), (235, 365), (235, 338), end_arrow=False, source="meteo")
    E((215, 449), (235, 449), (235, 365), end_arrow=False, source="music")
    E((360, 256), (360, 280), source="b_vol", target="b_main", label="seed", label_at=(384, 268))
    # bronze -> silver
    E((455, 138), (515, 138), source="b_wiki", target="s_wiki")
    E((455, 330), (485, 330), (485, 261), (515, 261), source="b_main", target="s_main")
    E((485, 330), (485, 433), (515, 433), target="s_events")
    E((605, 292), (605, 316), source="s_main", target="s_cats")
    # silver -> gold (one bus)
    E((695, 261), (725, 261), end_arrow=False, source="s_main")
    E((695, 347), (725, 347), end_arrow=False, source="s_cats")
    E((695, 433), (725, 433), end_arrow=False, source="s_events")
    for y, tgt in ((221, "g_outlook"), (299, "g_events"), (455, "g_narr"), (533, "g_ctx"), (631, "g_trend")):
        E((725, 261), (725, y), (755, y), target=tgt)
    # wiki -> vector search, straight across the empty top of the gold lane
    E((695, 138), (1025, 138), source="s_wiki", target="vs", label="Delta Sync (03c)")
    # gold -> serving
    for y, src in ((221, "g_outlook"), (299, "g_events"), (377, "g_lead"), (455, "g_narr"), (533, "g_ctx")):
        E((965, y), (995, y), end_arrow=False, source=src)
    E((995, 221), (995, 533), end_arrow=False)
    E((995, 386), (1025, 386), target="wh", label="UC reads", label_at=(995, 470))
    E((965, 631), (1025, 631), source="g_trend", target="lb", label="Synced Tables")
    # serving -> app
    E((1215, 138), (1275, 138), source="vs", target="agent", label="search")
    E((1215, 221), (1275, 221), source="fm", target="agent", start_arrow=True, label="chat")
    E((1215, 386), (1275, 386), source="wh", target="ui", label="SQL")
    E((1215, 608), (1275, 608), source="lb", target="ui", start_arrow=True, label="OLTP")
    E((1370, 260), (1370, 330), source="agent", target="ui", start_arrow=True, label="chat panel", label_at=(1370, 295))
    E((1495, 470), (1525, 470), source="ui", target="users", start_arrow=True)
    # analytics loop
    E((1120, 668), (1120, 828), source="lb", target="hist", label="Change Data Feed", label_at=(1120, 760))
    E((1025, 859), (965, 859), source="hist", target="events")
    E((755, 859), (695, 859), source="events", target="views")
    E((515, 859), (455, 859), source="views", target="dash")
    return p


# ---------------------------------------------------------------------------------------------
# Page 2: the nightly job
# ---------------------------------------------------------------------------------------------

def nightly() -> Page:
    p = Page("Nightly job", "nightly-job", 1540, 600, "The nightly pipeline",
             "One Lakeflow Job on serverless compute. Each notebook starts once everything it reads is fresh.")
    col = lambda i: 40 + 250 * i
    row = lambda j: 90 + 90 * j
    W, H = 210, 56
    nodes = [
        ("n02d", 0, 0, "02d · Wikipedia", "changed revisions only", "bronze"),
        ("n02b", 0, 1, "02b · Weather", "Open-Meteo observed + forecast", "bronze"),
        ("n02", 0, 2, "02 · Crime + 311", "SODA, watermarked appends", "bronze"),
        ("n02c", 0, 3, "02c · Events", "permits · setlist.fm · Ticketmaster", "bronze"),
        ("n03c", 1, 0, "03c · Wiki passages", "chunks + vector index sync", "silver"),
        ("n03", 1, 2, "03 · Silver", "type · clean · deduplicate", "silver"),
        ("n08", 2, 1, "08 · Next-week outlook", "backtest + live forecast", "gold"),
        ("n03b", 2, 2, "03b · Crime categories", "ai_classify, new pairs only", "silver"),
        ("n08b", 2, 3, "08b · Events look-ahead", "H3 · Gamma-Poisson", "gold"),
        ("n04", 3, 2, "04 · Gold trend tables", "windows · normals · profile", "gold"),
        ("n05", 4, 2, "05 · Leading indicators", "311 → crime lead-lag", "gold"),
        ("n09", 5, 1, "09 · Area narratives", "ai_query over the above", "gold"),
    ]
    for id, c, r, title, line, style in nodes:
        p.items.append(Box(id, col(c), row(r), W, H, title, (line,), style))
    p.items.append(Box("sync", col(4), row(3), W, H, "Re-trigger Synced Tables", ("Lakebase · manual step",), "manual",
                       dashed=True))
    mid = lambda r: row(r) + H / 2
    E = lambda *pts, **kw: p.items.append(Edge(list(pts), **kw))
    E((col(0) + W, mid(0)), (col(1), mid(0)), source="n02d", target="n03c")
    E((col(0) + W, mid(2)), (col(1), mid(2)), source="n02", target="n03")
    E((col(0) + W, mid(1)), (col(2), mid(1)), source="n02b", target="n08")
    E((col(0) + W, mid(3)), (col(2), mid(3)), source="n02c", target="n08b")
    E((col(1) + W, mid(2)), (col(2), mid(2)), source="n03", target="n03b")
    cx1, cx2, cx3 = col(1) + W / 2, col(2) + W / 2, col(3) + W / 2
    between12, between23 = row(1) + H + 17, row(2) + H + 17
    E((cx1, row(2)), (cx1, between12), (cx2, between12), (cx2, row(1) + H), source="n03", target="n08")
    E((cx1, row(2) + H), (cx1, between23), (cx2, between23), (cx2, row(3)), source="n03", target="n08b")
    E((col(2) + W, mid(2)), (col(3), mid(2)), source="n03b", target="n04")
    E((col(3) + W, mid(2)), (col(4), mid(2)), source="n04", target="n05")
    E((cx3, row(2) + H), (cx3, mid(3)), (col(4), mid(3)), source="n04", target="sync", dashed=True)
    gx = col(5) - 20
    E((col(4) + W, mid(2)), (gx, mid(2)), (gx, mid(1) + 10), (col(5), mid(1) + 10), source="n05", target="n09")
    E((col(2) + W, mid(1) - 10), (col(5), mid(1) - 10), source="n08", target="n09")
    p.items.append(Box("once", 40, 470, 1460, 76, "Run once, outside the job",
                       ("00 schema, volume, secrets   ·   01 / 01b full crime and 311 history   ·   01c seed weather and events "
                        "from the volume",
                        "06 / 07 analytics views over Lakebase CDF   ·   tests/smoke_test_agent_tools after changing tools "
                        "or connections"), "note"))
    return p


# ---------------------------------------------------------------------------------------------
# Page 3: the Lakebase data model
# ---------------------------------------------------------------------------------------------

def lakebase() -> Page:
    p = Page("Lakebase", "lakebase-model", 1340, 800, "Lakebase data model",
             "Postgres for the app's transactional reads and writes; Unity Catalog for everything analytical.")
    p.items.append(Lane(20, 70, 930, 700, "PUBLIC SCHEMA: THE APP'S TABLES", "serve"))
    p.items.append(Lane(980, 70, 340, 420, "SYNCED TABLES (READ-ONLY)", "gold"))
    p.items.append(Lane(980, 520, 340, 250, "UNITY CATALOG", "analytics"))
    T = lambda *a, **kw: p.items.append(Table(*a, **kw))
    T("subs", 40, 120, 260, "area_subscriptions", [("PK subscription_id", "bigserial"), ("FK user_id", "bigint"),
                                                    ("community_area", "int"), ("created_at", "timestamptz"),
                                                    ("unique (user_id, community_area)", "")])
    T("reports", 40, 330, 260, "resident_reports", [("PK report_id", "bigserial"), ("FK user_id", "bigint"),
                                                    ("community_area", "int"), ("description", "text"),
                                                    ("created_at", "timestamptz")])
    T("users", 360, 250, 230, "users", [("PK user_id", "bigserial"), ("email", "text, unique"), ("display_name", "text"),
                                       ("created_at", "timestamptz")])
    T("rules", 650, 110, 280, "alert_rules", [("PK rule_id", "bigserial"), ("FK user_id", "bigint"), ("community_area", "int"),
                                             ("metric", "text"), ("threshold_pct", "numeric"),
                                             ("kind", "'threshold' | 'trend'"), ("quiet_through", "date"),
                                             ("last_pct", "numeric"), ("created_at", "timestamptz")])
    T("tools", 650, 380, 280, "tool_invocations", [("PK invocation_id", "bigserial"), ("FK user_id", "bigint, null for reads"),
                                                   ("tool_name", "text"), ("community_area", "int"), ("success", "boolean"),
                                                   ("error_message", "text"), ("created_at", "timestamptz")])
    T("chat", 360, 450, 230, "chat_turns", [("PK turn_id", "bigserial"), ("FK user_id", "bigint"), ("chat_id", "text"),
                                           ("turn_index", "int"), ("active_area", "int"), ("user_message", "text"),
                                           ("assistant_message", "text"), ("tool_calls", "jsonb"), ("raw_messages", "jsonb"),
                                           ("model · latency_ms · error", ""), ("feedback", "-1 | 1"),
                                           ("feedback_at · created_at", "")])
    T("tm", 1000, 120, 300, "area_trend_metrics_lb", [("PK community_area", "int"), ("PK period", "date"),
                                                      ("PK metric", "text"), ("metric_count", "bigint"),
                                                      ("pct_change_vs_prior", "double"), ("rolling_3mo_avg", "double")],
      style="gold")
    T("tr", 1000, 300, 300, "area_trend_rolling_lb", [("PK community_area", "int"), ("PK metric", "text"),
                                                      ("PK window_days", "30 | 60 | 90"), ("window_count", "bigint"),
                                                      ("prior_window_count", "bigint"), ("as_of_date", "date")],
      style="gold")
    p.items.append(Box("gold", 1000, 570, 300, 56, "Gold: area_trend_metrics, _rolling", ("Delta, primary keys + CDF (04)",),
                       "gold"))
    p.items.append(Box("hist", 1000, 680, 300, 62, "lb_<table>_history", ("→ subscription_events, analytics views (06, 07)",),
                       "analytics"))
    E = lambda *pts, **kw: p.items.append(Edge(list(pts), **kw))
    E((300, 172), (330, 172), (330, 290), (360, 290), source="subs", target="users", label="user_id", label_at=(330, 230))
    E((300, 382), (330, 382), (330, 320), (360, 320), source="reports", target="users")
    E((650, 162), (620, 162), (620, 290), (590, 290), source="rules", target="users", label="user_id", label_at=(620, 225))
    E((650, 432), (620, 432), (620, 320), (590, 320), source="tools", target="users")
    E((475, 450), (475, 357), source="chat", target="users", label="user_id", label_at=(475, 410))
    E((1150, 570), (1150, 422), source="gold", target="tr", label="Synced Tables, triggered", label_at=(1150, 505))
    E((950, 711), (1000, 711), target="hist", label="CDF")
    return p


# ---------------------------------------------------------------------------------------------
# Page 4: one chat turn
# ---------------------------------------------------------------------------------------------

def agent() -> Page:
    p = Page("Agent", "agent-loop", 1340, 760, "One chat turn",
             "A plain tool-calling loop: the model picks tools, plain Python functions run them, the loop repeats until it answers.")
    p.items.append(Box("user", 40, 110, 170, 70, "Resident", ("chat panel, any view",), "user"))
    p.items.append(Box("route", 260, 110, 210, 70, "POST /chat", ("webapp/main.py",), "app"))
    p.items.append(Box("loop", 520, 90, 300, 110, "run_chat", ("src/agent_chat.py", "system prompt + metric names", "+ user id + the area in view",
                                                              "≤ 4 model rounds"), "app"))
    p.items.append(Box("fm", 900, 105, 230, 80, "Foundation Model API", ("Claude Sonnet", "OpenAI chat-completions protocol"),
                       "serve"))
    p.items.append(Lane(40, 260, 1260, 330, "TOOLS  ·  src/agent_tools.py, each wrapped in @_logged", "app"))
    tools = [
        ("get_area_trend", "Lakebase Synced Tables"), ("get_recent_activity", "silver tables, via SQL"),
        ("rank_areas", "all 77 areas at once"), ("get_leading_indicators", "indicator_lead_lag"),
        ("get_area_summary", "area_narratives"), ("get_next_week_outlook", "outlook_* tables"),
        ("get_upcoming_events", "events_upcoming"), ("search_area_background", "Vector Search, hybrid"),
    ]
    for i, (name, src) in enumerate(tools):
        p.items.append(Box(f"t{i}", 60 + 245 * (i % 4), 310 + 80 * (i // 4), 225, 56, name, (src,), "serve"))
    p.items.append(Box("w0", 60, 470, 225, 56, "subscribe_to_area", ("write · area_subscriptions",), "bronze"))
    p.items.append(Box("w1", 305, 470, 225, 56, "log_resident_report", ("write · resident_reports",), "bronze"))
    p.items.append(Box("log", 1050, 310, 230, 136, "Every call", ("logged to tool_invocations", "", "results go back to the",
                                                                   "model as JSON; errors as",
                                                                   "{ok: false, error}"), "note"))
    p.items.append(Box("toast", 570, 470, 440, 56, "Writes refresh the UI", ("toast + My Areas, through an HX-Trigger header",),
                       "note"))
    p.items.append(Box("turns", 260, 630, 300, 62, "chat_turns", ("question, tool trajectory, answer,", "latency, thumbs up/down"),
                       "analytics"))
    p.items.append(Box("answer", 640, 630, 300, 62, "The answer", ("Markdown, raw HTML escaped",), "app"))
    E = lambda *pts, **kw: p.items.append(Edge(list(pts), **kw))
    E((210, 145), (260, 145), source="user", target="route", start_arrow=True)
    E((470, 145), (520, 145), source="route", target="loop")
    E((820, 130), (900, 130), source="loop", target="fm", label="messages + tool specs", label_at=(860, 112))
    E((900, 160), (820, 160), source="fm", target="loop", label="tool calls or answer", label_at=(860, 178))
    E((620, 200), (620, 260), source="loop", label="tool calls", label_at=(620, 232))
    E((720, 260), (720, 200), target="loop", label="results", label_at=(720, 232))
    E((295, 180), (295, 630), source="route", target="turns", label="logged", label_at=(295, 610))
    E((540, 200), (540, 600), (790, 600), (790, 630), source="loop", target="answer", label="final answer", label_at=(665, 600))
    return p


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------

def _text_w(s: str, size: float, bold: bool = False) -> float:
    return len(s) * size * (0.56 if bold else 0.52)


def _check(page: Page) -> None:
    for i in page.items:
        if isinstance(i, Box):
            for k, (s, size, bold) in enumerate([(i.title, 13, True)] + [(ln, 11, False) for ln in i.lines]):
                if _text_w(s, size, bold) > i.w - 12:
                    print(f"  warning [{page.name}] {i.id}: {s!r} may overflow ({_text_w(s, size, bold):.0f} > {i.w - 12})")
            need = 22 + 15 * len(i.lines)
            if need > i.h:
                print(f"  warning [{page.name}] {i.id}: {len(i.lines)} lines need {need}px, box is {i.h}")


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def svg(page: Page) -> str:
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{page.width}" height="{page.height}" '
           f'viewBox="0 0 {page.width} {page.height}" font-family="{FONT}">',
           '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
           f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{EDGE}"/></marker></defs>',
           f'<rect width="100%" height="100%" fill="{PAGE}"/>',
           f'<text x="20" y="34" font-size="20" font-weight="700" fill="{INK}">{_esc(page.title)}</text>',
           f'<text x="20" y="55" font-size="12" fill="{INK_2}">{_esc(page.subtitle)}</text>']
    for i in page.items:
        if isinstance(i, Lane):
            _, stroke, fill, title = STYLES[i.style]
            out.append(f'<rect x="{i.x}" y="{i.y}" width="{i.w}" height="{i.h}" rx="10" fill="{fill}" stroke="{stroke}" '
                       f'stroke-opacity="0.35"/>')
            out.append(f'<text x="{i.x + 14}" y="{i.y + 22}" font-size="11" font-weight="700" letter-spacing="0.6" '
                       f'fill="{title}">{_esc(i.title)}</text>')
    for i in page.items:
        if isinstance(i, Edge):
            pts = " ".join(f"{x},{y}" for x, y in i.points)
            marks = (' marker-end="url(#arrow)"' if i.end_arrow else "") + (' marker-start="url(#arrow)"' if i.start_arrow else "")
            dash = ' stroke-dasharray="5 4"' if i.dashed else ""
            out.append(f'<polyline points="{pts}" fill="none" stroke="{EDGE}" stroke-width="1.5" '
                       f'stroke-linejoin="round"{dash}{marks}/>')
    for i in page.items:
        if isinstance(i, Box):
            fill, stroke, _, _ = STYLES[i.style]
            dash = ' stroke-dasharray="6 4"' if i.dashed else ""
            out.append(f'<rect x="{i.x}" y="{i.y}" width="{i.w}" height="{i.h}" rx="8" fill="{fill}" stroke="{stroke}" '
                       f'stroke-width="1.5"{dash}/>')
            block = 17 + 15 * len(i.lines)
            y0 = i.y + (i.h - block) / 2 + 13
            cx = i.x + i.w / 2
            out.append(f'<text x="{cx}" y="{y0}" font-size="13" font-weight="700" fill="{INK}" text-anchor="middle">'
                       f'{_esc(i.title)}</text>')
            for k, ln in enumerate(i.lines):
                out.append(f'<text x="{cx}" y="{y0 + 17 + 15 * k}" font-size="11" fill="{INK_2}" text-anchor="middle">'
                           f'{_esc(ln)}</text>')
        elif isinstance(i, Table):
            fill, stroke, _, _ = STYLES[i.style]
            out.append(f'<rect x="{i.x}" y="{i.y}" width="{i.w}" height="{i.h}" rx="6" fill="#ffffff" stroke="{stroke}" '
                       f'stroke-width="1.5"/>')
            out.append(f'<path d="M{i.x},{i.y + i.head_h} v-{i.head_h - 6} a6,6 0 0 1 6,-6 h{i.w - 12} a6,6 0 0 1 6,6 '
                       f'v{i.head_h - 6} z" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
            out.append(f'<text x="{i.x + 10}" y="{i.y + 19}" font-size="13" font-weight="700" fill="{INK}">{_esc(i.title)}</text>')
            for k, (name, typ) in enumerate(i.columns):
                y = i.y + i.head_h + 4 + i.row_h * k + 12
                key, name = (name[:2], name[3:]) if name[:3] in ("PK ", "FK ") else ("", name)
                if key:
                    out.append(f'<text x="{i.x + 10}" y="{y}" font-size="9" font-weight="700" fill="{stroke}">{key}</text>')
                out.append(f'<text x="{i.x + 30}" y="{y}" font-size="11" fill="{INK}">{_esc(name)}</text>')
                out.append(f'<text x="{i.x + i.w - 10}" y="{y}" font-size="10.5" fill="{MUTED}" text-anchor="end">{_esc(typ)}</text>')
        elif isinstance(i, Text):
            out.append(f'<text x="{i.x}" y="{i.y}" font-size="{i.size}" font-weight="{700 if i.bold else 400}" '
                       f'fill="{i.color}" text-anchor="{i.anchor}">{_esc(i.text)}</text>')
    for i in page.items:
        if isinstance(i, Edge) and i.label:
            x, y = i.label_at or _label_pos(i.points)
            w = _text_w(i.label, 10.5) + 10
            out.append(f'<rect x="{x - w / 2}" y="{y - 9}" width="{w}" height="17" rx="4" fill="{PAGE}"/>')
            out.append(f'<text x="{x}" y="{y + 4}" font-size="10.5" fill="{INK_2}" text-anchor="middle">{_esc(i.label)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _label_pos(points):
    segs = list(zip(points, points[1:]))
    (x1, y1), (x2, y2) = max(segs, key=lambda s: math.dist(*s))
    return (x1 + x2) / 2, (y1 + y2) / 2


def drawio(pages: list[Page]) -> str:
    def attr(**kw) -> str:
        return " ".join(f'{k}="{_esc(str(v))}"' for k, v in kw.items())

    def style(**kw) -> str:
        return ";".join(f"{k}={v}" for k, v in kw.items()) + ";"

    out = ['<mxfile host="build_diagrams.py" type="device">']
    for n, page in enumerate(pages):
        cells, boxes = [], {}
        cells.append(f'<mxCell {attr(id="title", value=f"<b>{_esc(page.title)}</b><br><font color={INK_2!r} style=\"font-size:12px\">{_esc(page.subtitle)}</font>", style=style(text=None, html=1, align="left", verticalAlign="top", whiteSpace="wrap", fontSize=20, fontColor=INK, fontFamily="Helvetica"), vertex=1, parent=1)}>'
                     f'<mxGeometry x="20" y="12" width="{page.width - 40}" height="50" as="geometry"/></mxCell>'.replace("text=None", "text"))
        for k, i in enumerate(page.items):
            if isinstance(i, Lane):
                _, stroke, fill, title = STYLES[i.style]
                cells.append(f'<mxCell {attr(id=f"lane{k}", value=i.title, style=style(rounded=1, arcSize=2, whiteSpace="wrap", html=1, fillColor=fill, strokeColor=stroke, strokeOpacity=35, verticalAlign="top", align="left", spacingLeft=12, spacingTop=4, fontStyle=1, fontSize=11, fontColor=title, fontFamily="Helvetica"), vertex=1, parent=1)}>'
                             f'<mxGeometry x="{i.x}" y="{i.y}" width="{i.w}" height="{i.h}" as="geometry"/></mxCell>')
        for k, i in enumerate(page.items):
            if isinstance(i, Box):
                fill, stroke, _, _ = STYLES[i.style]
                value = f"<b>{_esc(i.title)}</b>" + "".join(
                    f"<br><font style='font-size:11px' color='{INK_2}'>{_esc(ln) or '&nbsp;'}</font>" for ln in i.lines)
                boxes[i.id] = i
                cells.append(f'<mxCell {attr(id=i.id, value=value, style=style(rounded=1, arcSize=8, whiteSpace="wrap", html=1, fillColor=fill, strokeColor=stroke, strokeWidth=1.5, fontSize=13, fontColor=INK, fontFamily="Helvetica", dashed=int(i.dashed)), vertex=1, parent=1)}>'
                             f'<mxGeometry x="{i.x}" y="{i.y}" width="{i.w}" height="{i.h}" as="geometry"/></mxCell>')
            elif isinstance(i, Table):
                fill, stroke, _, _ = STYLES[i.style]
                boxes[i.id] = i
                cells.append(f'<mxCell {attr(id=i.id, value=i.title, style=style(swimlane=None, fontStyle=1, childLayout="stackLayout", horizontal=1, startSize=i.head_h, horizontalStack=0, resizeParent=1, resizeParentMax=0, resizeLast=0, collapsible=0, rounded=1, arcSize=4, html=1, fillColor=fill, strokeColor=stroke, strokeWidth=1.5, swimlaneFillColor="#ffffff", fontSize=13, fontColor=INK, fontFamily="Helvetica"), vertex=1, parent=1)}>'.replace("swimlane=None", "swimlane")
                             + f'<mxGeometry x="{i.x}" y="{i.y}" width="{i.w}" height="{i.h}" as="geometry"/></mxCell>')
                for r, (name, typ) in enumerate(i.columns):
                    key, nm = (name[:2], name[3:]) if name[:3] in ("PK ", "FK ") else ("", name)
                    value = (f"<b><font color='{stroke}' style='font-size:9px'>{key}</font></b>&nbsp;" if key else "") + _esc(nm) + \
                        (f"<font color='{MUTED}'>&nbsp;&nbsp;{_esc(typ)}</font>" if typ else "")
                    cells.append(f'<mxCell {attr(id=f"{i.id}_r{r}", value=value, style=style(text=None, html=1, align="left", verticalAlign="middle", spacingLeft=8, fontSize=11, fontColor=INK, fontFamily="Helvetica", strokeColor="none", fillColor="none"), vertex=1, parent=i.id)}>'.replace("text=None", "text")
                                 + f'<mxGeometry y="{i.head_h + i.row_h * r}" width="{i.w}" height="{i.row_h}" as="geometry"/></mxCell>')
        for k, i in enumerate(page.items):
            if not isinstance(i, Edge):
                continue
            st = dict(html=1, rounded=1, endArrow="block" if i.end_arrow else "none", endFill=1,
                      startArrow="block" if i.start_arrow else "none", startFill=1, strokeColor=EDGE, strokeWidth=1.5,
                      fontSize=10, fontColor=INK_2, labelBackgroundColor=PAGE, dashed=int(i.dashed), edgeStyle="none")
            ends = {}
            for role, end, pt in (("source", i.source, i.points[0]), ("target", i.target, i.points[-1])):
                b = boxes.get(end) if end else None
                if b is not None:
                    pre = "exit" if role == "source" else "entry"
                    st[f"{pre}X"] = round((pt[0] - b.x) / b.w, 4)
                    st[f"{pre}Y"] = round((pt[1] - b.y) / b.h, 4)
                    st[f"{pre}Perimeter"] = 0
                    ends[role] = end
            geo = (f'<mxGeometry relative="1" as="geometry">'
                   f'<mxPoint x="{i.points[0][0]}" y="{i.points[0][1]}" as="sourcePoint"/>'
                   f'<mxPoint x="{i.points[-1][0]}" y="{i.points[-1][1]}" as="targetPoint"/>'
                   + (f'<Array as="points">' + "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in i.points[1:-1]) + '</Array>'
                      if len(i.points) > 2 else "") + '</mxGeometry>')
            cells.append(f'<mxCell {attr(id=f"e{k}", value=i.label or "", style=style(**st), edge=1, parent=1, **ends)}>{geo}</mxCell>')
        out.append(f'<diagram id="page{n}" name="{_esc(page.name)}">'
                   f'<mxGraphModel dx="1200" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" '
                   f'fold="1" page="1" pageScale="1" pageWidth="{page.width}" pageHeight="{page.height}" math="0" shadow="0" '
                   f'background="{PAGE}">'
                   '<root><mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + '</root></mxGraphModel></diagram>')
    out.append("</mxfile>")
    return "\n".join(out)


def main() -> None:
    pages = [system(), nightly(), lakebase(), agent()]
    for page in pages:
        _check(page)
        (OUT / f"{page.file}.svg").write_text(svg(page))
    (OUT / "architecture.drawio").write_text(drawio(pages))
    print(f"wrote {len(pages)} SVGs and architecture.drawio to {OUT}")


if __name__ == "__main__":
    main()
