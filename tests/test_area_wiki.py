"""Offline tests for the Wikipedia background: the list-page and revision parsers and the chunker
in src/area_wiki.py, the search results' shaping in src/area_data.py, and the
search_area_background tool's argument checks. Nothing here calls Wikipedia or Databricks.

    python -m pytest tests/test_area_wiki.py
"""

from src import agent_tools as at
from src import area_data as ad
from src import area_wiki as aw

LIST_HTML = """
<p>intro</p>
<table class="wikitable sortable">
<tr><th>No.</th><th>Name</th><th>Population</th></tr>
<tr><th><span class="nobold">(sq mi.)</span></th><th>(km2)</th></tr>
<tr><td>01</td><td><a href="/wiki/Rogers_Park,_Chicago" title="Rogers Park, Chicago">Rogers Park</a></td><td>54,173</td></tr>
<tr><td>32</td><td><a href="/wiki/Loop,_Chicago#History">(The) Loop</a><sup>[a]</sup></td><td>42,298</td></tr>
<tr><td>76</td><td><a href="/wiki/O%27Hare,_Chicago">O&#39;Hare</a></td><td>13,418</td></tr>
<tr><td>Total</td><td>Chicago<sup>[12]</sup></td><td>2,711,226</td></tr>
</table>
<table class="wikitable"><tr><td>99</td><td><a href="/wiki/Other">Other</a></td></tr></table>
"""


def test_parse_area_table_reads_the_first_table_only():
    areas = aw.parse_area_table(LIST_HTML)
    assert areas == {
        1: {"name": "Rogers Park", "title": "Rogers Park, Chicago"},
        32: {"name": "(The) Loop", "title": "Loop, Chicago"},
        76: {"name": "O'Hare", "title": "O'Hare, Chicago"},
    }


def test_parse_revisions_follows_normalization_and_redirects():
    body = {"query": {
        "normalized": [{"from": "Loop,_Chicago", "to": "Loop, Chicago"}],
        "redirects": [{"from": "Loop, Chicago", "to": "Chicago Loop"}],
        "pages": [{"title": "Chicago Loop", "revisions": [{"revid": 111}]},
                  {"title": "Rogers Park, Chicago", "revisions": [{"revid": 222}]},
                  {"title": "Gone, Chicago", "missing": True}],
    }}
    revs = aw.parse_revisions(body)
    assert revs["Loop,_Chicago"] == 111
    assert revs["Loop, Chicago"] == 111
    assert revs["Rogers Park, Chicago"] == 222
    assert "Gone, Chicago" not in revs


EXTRACT = """Albany Park is a community area.
It is on the Northwest Side.

== History ==
Settled in the 1890s.

=== Early years ===
The Ravenswood line arrived in 1907.

==== The terminal ====
Kimball is the end of the line.

== Neighborhoods ==

=== Mayfair ===
Mayfair sits to the west.

== See also ==
List of things

== References ==
Ref one.
"""


def test_split_sections_labels_skips_and_folds():
    sections = aw.split_sections(EXTRACT)
    assert sections == [
        ("Overview", ["Albany Park is a community area.", "It is on the Northwest Side."]),
        ("History", ["Settled in the 1890s."]),
        ("History > Early years", ["The Ravenswood line arrived in 1907.", "Kimball is the end of the line."]),
        ("Neighborhoods > Mayfair", ["Mayfair sits to the west."]),
    ]


def _para(n: int, word: str = "word") -> str:
    return " ".join([word] * (n - 1)) + " end."


def test_chunk_article_packs_within_sections_with_overlap():
    extract = "\n".join([_para(30, "lead"), "== History ==", _para(200, "a"), _para(100, "b"), _para(150, "c"),
                         "== Politics ==", "Too short to keep."])
    rows = aw.chunk_article(14, "Albany Park", extract, "https://x", 7, target_words=350)
    assert [(r["chunk_id"], r["section"]) for r in rows] == [("14-0", "Overview"), ("14-1", "History"), ("14-2", "History")]
    first, second = rows[1]["content"], rows[2]["content"]
    assert first.startswith("Albany Park (Chicago community area 14) - History\n\n")
    assert rows[1]["word_count"] == 300
    # the 100-word "b" paragraph closes the first History chunk and is repeated to open the second
    assert aw.passage_text(second).startswith(_para(100, "b"))
    assert rows[2]["word_count"] == 250
    assert all(r["revid"] == 7 and r["page_url"] == "https://x" for r in rows)


def test_chunk_article_splits_an_overlong_paragraph():
    long_para = " ".join(_para(100, f"s{i}") for i in range(9))    # nine 100-word sentences
    rows = aw.chunk_article(1, "Rogers Park", "== History ==\n" + long_para, "u", 1, target_words=350)
    assert [r["word_count"] for r in rows] == [300, 300, 300]


def test_shape_wiki_hits_casts_and_strips_the_header():
    columns = ["community_area", "area_name", "section", "content", "page_url", "score"]
    rows = [["24.0", "West Town", "Neighborhoods", "West Town (Chicago community area 24) - Neighborhoods\n\nWicker Park is here.",
             "https://en.wikipedia.org/wiki/West_Town,_Chicago", "0.03278689"]]
    assert ad.shape_wiki_hits(columns, rows) == [{
        "community_area": 24, "area_name": "West Town", "section": "Neighborhoods", "text": "Wicker Park is here.",
        "page_url": "https://en.wikipedia.org/wiki/West_Town,_Chicago", "score": 0.033,
    }]


def test_search_area_background_checks_arguments(monkeypatch):
    calls = []
    monkeypatch.setattr(ad, "search_wiki", lambda q, a, n: calls.append((q, a, n)) or [])
    monkeypatch.setattr(at, "_log_invocation", lambda *a, **k: None)
    assert at.search_area_background("  ")["ok"] is False
    assert at.search_area_background("history", community_area=78)["ok"] is False
    out = at.search_area_background(" history ", community_area=24, num_results=50)
    assert out["ok"] and out["area_name"] == "West Town" and out["source"] == aw.LICENSE
    assert calls == [("history", 24, 10)]
