"""Wikipedia background for the 77 community areas: fetching the articles, and cutting them into
the passages the agent's search_area_background tool retrieves.

Feeds `02d_bronze_wikipedia` (fetch) and `03c_wiki_chunks` (chunk, then the vector search index).
Uses the MediaWiki Action API rather than scraping article HTML: it returns plain text with the
section headings kept as `== History ==` lines, plus each article's revision id, which is the
watermark. The one HTML parse is the list page's first table, which maps area numbers to article
titles.

Wikipedia asks API clients for a descriptive User-Agent and serial requests; 77 articles once is
well inside that. Text is CC BY-SA, so every passage keeps its article URL for attribution.

Pure Python (requests only). The parsers and the chunker do no I/O, so the tests cover them
offline (`tests/test_area_wiki.py`).
"""

from __future__ import annotations

import html
import re
import time
from datetime import datetime
from typing import Any
from urllib.parse import quote, unquote

import requests

API = "https://en.wikipedia.org/w/api.php"
LIST_PAGE = "Community_areas_of_Chicago"
USER_AGENT = "chicago-crime-early-warning/1.0 (https://github.com/ajwaters09/chicago-crime-early-warning)"
LICENSE = "Wikipedia, CC BY-SA 4.0"

# Sections that are lists of links or citations, not prose worth retrieving.
SKIP_SECTIONS = {"references", "see also", "external links", "notes", "further reading", "works cited",
                 "bibliography", "sources", "citations", "footnotes", "gallery", "notes and references"}
LEAD_SECTION = "Overview"

TARGET_WORDS = 350      # a chunk closes once it reaches about this many words
OVERLAP_MAX_WORDS = 120  # a new chunk in the same section repeats the previous paragraph if it's this short
MIN_WORDS = 20          # shorter chunks (a one-line section) are dropped

_session: requests.Session | None = None


def _get(params: dict[str, Any], tries: int = 3) -> dict[str, Any]:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    for attempt in range(tries):
        resp = _session.get(API, params={"format": "json", "formatversion": 2, **params}, timeout=60)
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < tries - 1:
            time.sleep(2 ** attempt * 2)
            continue
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise RuntimeError(f"Wikipedia API error: {body['error']}")
        return body
    raise RuntimeError("unreachable")


def page_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"), safe="(),'")


# ---------------------------------------------------------------------------
# The list page: area number -> article title
# ---------------------------------------------------------------------------


def _cell_text(cell_html: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", cell_html)).strip()


def parse_area_table(page_html: str) -> dict[int, dict[str, str]]:
    """The list page's first wikitable -> {area number: {"name": ..., "title": ...}}.

    Rows whose first cell isn't a number (the header rows, the citywide total) are skipped.
    `title` is the linked article's title, e.g. "Rogers Park, Chicago"; some are redirects, which
    the article fetch resolves.
    """
    table = re.search(r'<table class="wikitable.*?</table>', page_html, re.S)
    if not table:
        raise ValueError("no wikitable on the community areas page")
    out: dict[int, dict[str, str]] = {}
    for row in re.findall(r"<tr.*?</tr>", table.group(0), re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        if len(cells) < 2 or not _cell_text(cells[0]).isdigit():
            continue
        link = re.search(r'href="/wiki/([^"#]+)[^"]*"', cells[1])
        if not link:
            continue
        out[int(_cell_text(cells[0]))] = {
            "name": re.sub(r"\[[^\]]*\]", "", _cell_text(cells[1])).strip(),   # drop footnote marks
            "title": unquote(link.group(1)).replace("_", " "),
        }
    return out


def fetch_area_titles() -> dict[int, dict[str, str]]:
    body = _get({"action": "parse", "page": LIST_PAGE, "prop": "text"})
    return parse_area_table(body["parse"]["text"])


# ---------------------------------------------------------------------------
# Revisions (the watermark) and article text
# ---------------------------------------------------------------------------


def parse_revisions(body: dict[str, Any]) -> dict[str, int]:
    """A batched `prop=revisions` response -> {requested title: latest revid}.

    The API answers under each page's final title, after normalization ("Loop, Chicago") and
    redirects ("Chicago Loop"), and lists both steps separately; this follows them back to the
    title that was asked for.
    """
    query = body.get("query", {})
    final_revid = {p["title"]: p["revisions"][0]["revid"]
                   for p in query.get("pages", []) if p.get("revisions")}
    steps = {s["from"]: s["to"] for s in query.get("normalized", []) + query.get("redirects", [])}

    def resolve(title: str) -> str:
        seen = set()
        while title in steps and title not in seen:
            seen.add(title)
            title = steps[title]
        return title

    requested = set(steps) | set(final_revid)
    return {t: final_revid[resolve(t)] for t in requested if resolve(t) in final_revid}


def fetch_revisions(titles: list[str]) -> dict[str, int]:
    """Latest revid per title, 50 titles a request (the API's batch limit)."""
    out: dict[str, int] = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        body = _get({"action": "query", "prop": "revisions", "rvprop": "ids",
                     "titles": "|".join(batch), "redirects": 1})
        revs = parse_revisions(body)
        out.update({t: revs[t] for t in batch if t in revs})
    return out


def fetch_article(title: str) -> dict[str, Any]:
    """One article's plain text and revision. `title` is the final title, after redirects."""
    body = _get({"action": "query", "prop": "extracts|revisions", "explaintext": 1,
                 "exsectionformat": "wiki", "rvprop": "ids|timestamp",
                 "titles": title, "redirects": 1})
    page = body["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        raise ValueError(f"no article text for {title!r}")
    rev = page["revisions"][0]
    return {
        "title": page["title"],
        "page_url": page_url(page["title"]),
        "revid": int(rev["revid"]),
        "rev_timestamp": datetime.fromisoformat(rev["timestamp"].replace("Z", "+00:00")),
        "extract": page["extract"],
    }


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

_HEADING = re.compile(r"^(={2,})\s*(.+?)\s*\1\s*$")


def split_sections(extract: str) -> list[tuple[str, list[str]]]:
    """Plain-text extract -> [(section label, paragraphs)], in article order.

    The lead is "Overview". Labels chain the top two heading levels ("History > Early
    settlement"); deeper headings fold into their level-3 parent. Anything under a SKIP_SECTIONS
    heading is dropped, and so are sections with no text of their own.
    """
    sections: list[tuple[str, list[str]]] = []
    path: list[str] = []
    label, paras = LEAD_SECTION, []

    def close() -> None:
        if paras:
            sections.append((label, list(paras)))

    for line in extract.splitlines():
        m = _HEADING.match(line.strip())
        if m:
            level = len(m.group(1))
            if level > 3:
                if len(path) >= 2:
                    continue   # a level-4+ heading keeps writing into its level-3 parent
                level = 3
            close()
            paras = []
            path = path[:level - 2] + [m.group(2)]
            label = " > ".join(path)
            continue
        text = line.strip()
        if text:
            paras.append(text)
    close()
    return [(lbl, ps) for lbl, ps in sections if lbl.split(" > ")[0].lower() not in SKIP_SECTIONS]


def _words(text: str) -> int:
    return len(text.split())


def _split_long(paragraph: str, limit: int) -> list[str]:
    """A paragraph longer than `limit` words -> pieces of at most `limit` words, cut at sentences
    where it can be (at words, for a single overlong sentence)."""
    if _words(paragraph) <= limit:
        return [paragraph]
    pieces, cur = [], []
    for sentence in re.split(r"(?<=[.!?])\s+", paragraph):
        words = sentence.split()
        while len(words) > limit:
            if cur:
                pieces.append(" ".join(cur))
                cur = []
            pieces.append(" ".join(words[:limit]))
            words = words[limit:]
        if cur and len(cur) + len(words) > limit:
            pieces.append(" ".join(cur))
            cur = []
        cur += words
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_article(community_area: int, area_name: str, extract: str, page_url: str, revid: int,
                  target_words: int = TARGET_WORDS) -> list[dict[str, Any]]:
    """One article -> the rows of `silver_wiki_chunks`.

    Paragraphs pack into chunks of about `target_words`, never across sections, so each chunk is
    about one thing. A new chunk within a section starts by repeating the previous paragraph when
    that's short, so a fact split across the boundary is still retrievable. Each chunk's `content`
    opens with the area and section, so a short chunk still says where it's from, both to the
    embedding and to the model reading it.

    chunk_id is "<area>-<n>": stable within a revision. A changed article gets all its chunks
    replaced, so ids needn't survive edits.
    """
    rows: list[dict[str, Any]] = []
    header_area = f"{area_name} (Chicago community area {community_area})"
    for section, paras in split_sections(extract):
        pieces = [p for para in paras for p in _split_long(para, target_words)]
        chunks: list[list[str]] = []
        cur: list[str] = []
        for p in pieces:
            if cur and sum(map(_words, cur)) + _words(p) > target_words:
                chunks.append(cur)
                cur = [cur[-1]] if _words(cur[-1]) <= OVERLAP_MAX_WORDS else []
            cur.append(p)
        if cur:
            chunks.append(cur)
        for chunk in chunks:
            body = "\n\n".join(chunk)
            if _words(body) < MIN_WORDS:
                continue
            rows.append({
                "chunk_id": f"{community_area}-{len(rows)}",
                "community_area": community_area,
                "area_name": area_name,
                "section": section,
                "chunk_index": len(rows),
                "content": f"{header_area} - {section}\n\n{body}",
                "word_count": _words(body),
                "page_url": page_url,
                "revid": revid,
            })
    return rows


def passage_text(content: str) -> str:
    """`content` without its "<area> - <section>" header line (the tool returns those as fields)."""
    return content.split("\n\n", 1)[1] if "\n\n" in content else content
