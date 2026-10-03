"""The chat agent: agent_tools.py's tools behind an LLM tool-calling loop.

Databricks Foundation Model APIs speak the OpenAI chat-completions protocol, so the standard
`openai` client is pointed at the workspace's `/serving-endpoints` route, authenticated as
whatever identity runs the code (the app's service principal when deployed). No agent framework:
one loop over plain Python functions is all a chat panel needs, and every tool stays an ordinary
@_logged function in agent_tools.py.

To add a tool: write it in agent_tools.py with @_logged, describe it in TOOL_SPECS, add it to
_DISPATCH, and mention when to use it in SYSTEM_PROMPT. tests/test_agent_wiring.py checks the
three agree.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

from src import agent_tools, config
from src.community_areas import area_name

SERVING_ENDPOINT = config.SERVING_ENDPOINT

SYSTEM_PROMPT = (
    "You are the Neighborhood Watch assistant for a Chicago crime and 311 "
    "early-warning app. You help residents look up crime/311 trends for "
    "their community area, subscribe to trend updates, and file concern "
    "reports. Community areas are numbered 1-77. Always confirm which "
    "community area a user means before calling a tool if it's ambiguous. "
    "There is no email/SMS/push delivery -- subscribing only means trend "
    "changes will show up in this app's My Areas tab next "
    "time the user opens it. Never tell a user they'll be notified, "
    "emailed, or alerted outside the app. "
    "Reply once per turn: when you call tools, call them without any text alongside (no 'let me "
    "look that up' or restating the question), and write your whole answer after you have what "
    "you need. The app shows only that final answer. "
    "Every area number has a baseline; always name it. get_area_trend gives each metric "
    "pct_vs_normal (the window vs. what the area's own past year predicts for this time of "
    "year -- the same 'normal' the week-ahead forecast uses) and pct_change_vs_prior_window "
    "(the window vs. the same-length window right before it). For 'is it getting better or "
    "worse', lead with pct_vs_normal and give pct_change_vs_prior_window as the short-term "
    "swing. When they disagree, explain it: e.g. 'up 34% from an unusually quiet stretch, but "
    "only about 10% above normal'. Its usual_level says where the area sits over the last 12 months: "
    "violent and all crime per 1,000 residents a year, its rank of 77 and a label ('among the highest'). "
    "Give it for context (a busy area running below normal is still a busy area), and note that "
    "downtown areas' per-resident rates run high because of visitors. "
    "Be even-handed: say plainly when an area is above normal or rising, and when it's below; "
    "don't soften bad news or talk up good news, and don't be alarmist either. "
    "pct_change_vs_prior (month-over-month) "
    "can look artificially good early in a calendar month, since a partial "
    "month is being compared against a full prior one; only lean on it for "
    "describing the longer month-by-month shape via `history`, not as the "
    "headline 'better or worse' number. "
    "In get_area_trend's monthly history, the latest month is usually "
    "still in progress: an entry with partial: true also carries "
    "projected_count, a straight-line projection to a full month. Say "
    "'on pace for about N' rather than treating the partial count as a "
    "real drop. "
    "For questions that compare or rank areas ('which areas have the "
    "biggest increase in crime?', 'where is graffiti worst?'), call "
    "rank_areas once instead of get_area_trend per area. For 'getting worse' "
    "questions prefer sort_by above_normal (steadier than pct_increase, which "
    "rewards areas coming off a quiet stretch); for 'most dangerous' or "
    "'safest' areas use highest_usual_level / lowest_usual_level on "
    "crime_violent. Percentage rankings skip areas with a small baseline "
    "(min_prior_count) so a jump from 2 to 4 incidents doesn't top the "
    "list; mention that, and give absolute counts alongside percentages. "
    "For 'what's going on in X?' start with get_area_summary, then add "
    "trend numbers if useful. get_leading_indicators reports correlations "
    "between earlier 311 requests and later crime; always describe them as "
    "historical patterns worth watching, never as proof that one causes "
    "the other or as a prediction for a specific place. Its `watch` list "
    "is leading 311 types rising in the area now; `near` is ones climbing "
    "but not there yet. "
    "get_next_week_outlook is a forecast for this week (Monday to Sunday), not an observed "
    "trend: 'up' or 'down' means likely above or below the area's normal for this time of year, "
    "not versus last week, and 'unclear' means too close to call. Say it's a forecast, quote its "
    "track_record (e.g. 'right about 75% of the time when it makes a call'), and keep its two "
    "parts apart: citywide_part_pct (weather, the city's recent trend, holidays -- the same for "
    "every area) and local_part_pct (this area's own lean). A 'below normal' call driven by a cool "
    "week citywide says nothing good about the area itself; if the area has been running above "
    "normal lately (get_area_trend's pct_vs_normal), say so alongside the forecast. Its `effects` "
    "hold the model's fitted effects (per degree F warmer than normal, per inch of rain, holiday "
    "weeks, momentum); use them for questions like 'does weather affect crime?'. "
    "get_upcoming_events lists street festivals and club nights ahead and the extra incidents "
    "they tend to bring to the blocks around them. The numbers are small (often under one), so "
    "say 'about N extra incidents nearby over the event', never call an event dangerous, and "
    "mention that pub crawls and big street closures are where most of it lands. "
    "search_area_background searches passages from each community area's Wikipedia article: history, "
    "the neighborhoods inside it, landmarks, parks, schools, transit and politics. Use it for 'tell me "
    "about X', history and 'what's it like there' questions, and search without community_area to find "
    "which area a neighborhood or landmark is in (Wicker Park, for instance, is in West Town). Say the "
    "background comes from Wikipedia and link the article with a markdown link to its page_url. The "
    "articles' population, demographic and crime figures are often a decade or more old: never use them "
    "for how much crime an area has now or which way it's heading (the other tools own those), and don't "
    "let an article's reputation for crime color how you read the current numbers. "
    "Use imperial units: degrees Fahrenheit, inches, miles. "
    "When a tool returns ok: false, tell the user what went wrong in plain "
    "language rather than surfacing the raw error."
)

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format) and the dispatch table
# ---------------------------------------------------------------------------

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_area_trend",
            "description": (
                "Current crime/311 trend scores for a Chicago community area: monthly history for "
                "charting the shape over time; a 30/60/90-day window compared with normal for this "
                "time of year (pct_vs_normal, the steadier 'better or worse' signal) and with the "
                "window right before it (pct_change_vs_prior_window, the short-term swing); and the "
                "area's usual_level per resident with its citywide rank."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "community_area": {"type": "integer", "description": "Chicago community area number, 1-77."},
                    "months": {"type": "integer", "description": "Months of monthly history per metric (default 6)."},
                    "window_days": {
                        "type": "integer",
                        "enum": [30, 60, 90],
                        "description": (
                            "Trailing-window size for the apples-to-apples comparison (default 30). "
                            "Use 60 or 90 if the user wants a longer-term read, or asked about the last "
                            "couple months rather than just recently."
                        ),
                    },
                },
                "required": ["community_area"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_areas",
            "description": (
                "Rank all 77 community areas on one metric in a single call: trailing-window change, "
                "vs. normal for this time of year, volume, or usual level per resident. Use for any "
                "cross-area question: getting worse, biggest increases, most/least crime per resident. "
                "Percentage sorts skip areas below min_prior_count so tiny baselines don't dominate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": (
                            "Metric name -- use one from the list in the system prompt, e.g. crime_total, "
                            "crime_fraud_financial, 311_graffiti. Crime categories: violent, property, drugs, "
                            "weapons, public_order, fraud_financial, other."
                        ),
                    },
                    "window_days": {"type": "integer", "enum": [30, 60, 90], "description": "Trailing window (default 30)."},
                    "sort_by": {
                        "type": "string",
                        "enum": list(agent_tools.RANK_SORTS),
                        "description": (
                            "pct_increase (default) / pct_decrease: vs. the prior window. abs_increase / "
                            "abs_decrease: counts. volume: current-window count. above_normal / "
                            "below_normal: vs. normal for this time of year (best for 'getting worse'). "
                            "highest_usual_level / lowest_usual_level: usual level per 1,000 residents "
                            "(the last 12 months; best for 'most dangerous' / 'safest')."
                        ),
                    },
                    "limit": {"type": "integer", "description": "How many areas to return (default 10)."},
                    "min_prior_count": {"type": "integer", "description": "Minimum prior-window count for pct sorts (default 20)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_leading_indicators",
            "description": (
                "Which 311 service-request types have historically risen 1-3 months BEFORE specific crime "
                "categories in the same area (a lagged correlation across all areas, controlling for area "
                "size and citywide/seasonal swings). With community_area, also lists which of those 311 "
                "types are rising there right now -- early-warning signals to watch."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "community_area": {"type": "integer", "description": "Optional community area number, 1-77."},
                    "window_days": {"type": "integer", "enum": [30, 60, 90], "description": "Window for the 'rising now' check (default 30)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_area_summary",
            "description": (
                "A short plain-language summary of what's been happening in a community area lately "
                "(recent crime and 311 patterns, hotspot blocks, rising signals), with the structured "
                "facts it was written from. Good first call for 'what's going on in X?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "community_area": {"type": "integer", "description": "Chicago community area number, 1-77."},
                },
                "required": ["community_area"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_next_week_outlook",
            "description": (
                "This week's crime forecast: whether a community area is likely to come in above or below "
                "its normal for this time of year (call up/down/unclear with a probability), the citywide "
                "outlook and what drives it (recent momentum, weather forecast, holidays), which areas are "
                "called up or down, and the model's backtested track record. Use for 'what's this week "
                "looking like?' or 'is crime expected to go up in X?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "community_area": {"type": "integer", "description": "Optional community area number, 1-77."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_upcoming_events",
            "description": (
                "Street festivals (from filed city permits) and club nights (ticketed shows at 20 small "
                "music clubs) coming up, with the extra crime each tends to bring to the blocks around it, "
                "based on past editions and similar events. Use for 'what's happening near X?', 'any "
                "events coming up?' or 'why might this weekend be busy?'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "community_area": {"type": "integer", "description": "Optional community area number, 1-77."},
                    "days": {"type": "integer", "description": "How many days ahead to look, 1-90 (default 30)."},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_area_background",
            "description": (
                "Search Wikipedia background on Chicago's community areas: history, the neighborhoods "
                "inside each area, landmarks, parks, schools, transit, politics, notable people. Returns "
                "the best-matching passages with their article links. Leave out community_area to search "
                "all 77, e.g. to find which area a neighborhood or landmark is in."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look for, e.g. 'history', 'Wicker Park', 'parks and lakefront'.",
                    },
                    "community_area": {"type": "integer", "description": "Optional community area number, 1-77."},
                    "num_results": {"type": "integer", "description": "Passages to return, 1-10 (default 5)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_activity",
            "description": "Recent individual crime and 311 records for a Chicago community area.",
            "parameters": {
                "type": "object",
                "properties": {
                    "community_area": {"type": "integer", "description": "Chicago community area number, 1-77."},
                    "days": {"type": "integer", "description": "How many days back to look (default 30)."},
                    "limit": {"type": "integer", "description": "Max combined records to return (default 25)."},
                },
                "required": ["community_area"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "subscribe_to_area",
            "description": "Subscribe a user to trend updates for a community area. Idempotent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "Existing users.user_id."},
                    "community_area": {"type": "integer", "description": "Chicago community area number, 1-77."},
                },
                "required": ["user_id", "community_area"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_resident_report",
            "description": "File a resident concern report for a community area.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "integer", "description": "Existing users.user_id."},
                    "community_area": {"type": "integer", "description": "Chicago community area number, 1-77."},
                    "description": {"type": "string", "description": "Free-text description of the concern."},
                },
                "required": ["user_id", "community_area", "description"],
            },
        },
    },
]

_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "get_area_trend": agent_tools.get_area_trend,
    "get_recent_activity": agent_tools.get_recent_activity,
    "rank_areas": agent_tools.rank_areas,
    "get_leading_indicators": agent_tools.get_leading_indicators,
    "get_area_summary": agent_tools.get_area_summary,
    "get_next_week_outlook": agent_tools.get_next_week_outlook,
    "get_upcoming_events": agent_tools.get_upcoming_events,
    "search_area_background": agent_tools.search_area_background,
    "subscribe_to_area": agent_tools.subscribe_to_area,
    "log_resident_report": agent_tools.log_resident_report,
}


def _get_client():
    """An OpenAI client for this workspace's Foundation Model APIs.

    The bearer token comes from `Config.authenticate()`, which works for any configured auth
    method (OAuth here), unlike `config.token` (PATs only) or `config.oauth_token()` (OAuth only).
    """
    from databricks.sdk import WorkspaceClient
    from openai import OpenAI

    cfg = WorkspaceClient().config
    bearer_token = cfg.authenticate()["Authorization"].removeprefix("Bearer ")
    return OpenAI(base_url=f"{cfg.host}/serving-endpoints", api_key=bearer_token)


def _run_tool_call(tool_call) -> dict[str, Any]:
    name = tool_call.function.name
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"ok": False, "error": f"unknown tool: {name}"}
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"bad tool arguments: {exc}"}
    try:
        return fn(**args)
    except Exception as exc:
        return {"ok": False, "error": f"tool call failed: {exc}"}


def run_chat(
    messages: list[dict[str, Any]],
    max_tool_rounds: int = 4,
    current_user_id: int | None = None,
    active_area: int | None = None,
    trace: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run one user turn through the tool-calling loop.

    Args:
        messages: the conversation so far, as OpenAI chat-format dicts, without the system
            prompt (this prepends it). Not mutated.
        max_tool_rounds: model calls allowed in one turn.
        current_user_id: the logged-in user, added to the system prompt so the model can fill
            `user_id` on the write tools without asking.
        active_area: the area open in the app, so "how's it trending here?" resolves.
        trace: if given, filled in for the chat_turns log: model, total latency_ms, and each tool
            call's name, arguments, ok/error and latency_ms.

    Returns:
        The new assistant and tool messages to append to the conversation.
    """
    started = time.monotonic()
    tool_log: list[dict[str, Any]] = []
    if trace is not None:
        trace.update({"model": SERVING_ENDPOINT, "tool_calls": tool_log})
    client = _get_client()
    system_content = SYSTEM_PROMPT
    if current_user_id is not None:
        system_content += f" The current logged-in user's user_id is {current_user_id}."
    # The real metric names, so the model doesn't guess. Best-effort: without them the tools
    # still resolve loose names and list the valid ones on a miss.
    try:
        system_content += (
            " Metric names available to the tools (use these exact strings): "
            + ", ".join(agent_tools.list_metric_names()) + "."
        )
    except Exception:
        pass
    if active_area is not None:
        system_content += (
            f" The user is currently viewing {area_name(active_area)} (community area {active_area}) "
            "in the app; 'here' or 'this area' means that one."
        )
    full = [{"role": "system", "content": system_content}, *messages]
    new_messages: list[dict[str, Any]] = []

    for _ in range(max_tool_rounds):
        response = client.chat.completions.create(
            model=SERVING_ENDPOINT,
            messages=full,
            tools=TOOL_SPECS,
        )
        choice = response.choices[0].message
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": choice.content or ""}
        if choice.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.tool_calls
            ]
        full.append(assistant_msg)
        new_messages.append(assistant_msg)

        if not choice.tool_calls:
            break

        for tc in choice.tool_calls:
            tool_started = time.monotonic()
            result = _run_tool_call(tc)
            tool_log.append({
                "name": tc.function.name,
                "arguments": tc.function.arguments,
                "ok": bool(result.get("ok")),
                "error": result.get("error"),
                "latency_ms": int((time.monotonic() - tool_started) * 1000),
            })
            # default=str: tool results can carry dates and Decimals from Postgres.
            tool_msg = {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result, default=str)}
            full.append(tool_msg)
            new_messages.append(tool_msg)

    if trace is not None:
        trace["latency_ms"] = int((time.monotonic() - started) * 1000)
    return new_messages
