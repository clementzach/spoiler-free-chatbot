import json
from types import SimpleNamespace as NS

import pytest

from bench.tools import StubWiki, check, load_tool_cases, run_trial


def test_cases_load_and_are_well_formed():
    cases = load_tool_cases()
    assert len(cases) >= 10
    for c in cases:
        assert c["context"]["progress"] and c["context"]["user_message"]
        assert "wiki" in c and c["expect"]


@pytest.mark.asyncio
async def test_stub_serves_intro_full_and_missing():
    s = StubWiki({"search": "hits", "pages": {"Dune": {"intro": "i", "full": "f"}}})
    assert await s("wikipedia_search", '{"query": "x"}') == "hits"
    assert await s("wikipedia_page", '{"title": "dune"}') == "i"
    assert await s("wikipedia_page", '{"title": "Dune", "full": true}') == "f"
    assert "No article" in await s("wikipedia_page", '{"title": "Nope"}')
    assert s.calls[1] == ("wikipedia_page", {"title": "dune"})
    assert (await StubWiki({"error": True})("wikipedia_search", "{}")).startswith("Lookup failed")


def test_check_each_expectation():
    search = [("wikipedia_search", {"query": "Dune novel author"})]
    assert check({"lookup": "required", "query_mentions": ["dune"]}, search, "x") == []
    assert check({"lookup": "required"}, [], "x")
    assert check({"lookup": "forbidden"}, search, "x")
    assert check({"query_mentions": ["Arrival"]}, search, "x")
    full = [("wikipedia_page", {"title": "Dune", "full": True})]
    assert check({"no_full": True}, full, "x") and not check({"no_full": True}, search, "x")
    assert check({"page_fetched": True}, search, "x")
    assert check({"reply_contains_any": ["Herbert"]}, [], "Frank herbert") == []
    assert check({"reply_contains_any": ["Herbert"]}, [], "nobody")
    assert check({"reply_excludes": ["train"]}, [], "A Train!")


@pytest.mark.asyncio
async def test_run_trial_end_to_end_with_fake_model():
    case = {
        "context": {"progress": "Dune, ch. 3", "user_message": "Who wrote it?"},
        "wiki": {"search": "Dune (novel): by Frank Herbert"},
        "expect": {"lookup": "required", "query_mentions": ["Dune"], "reply_contains_any": ["Herbert"]},
    }
    call = NS(id="c1", function=NS(name="wikipedia_search", arguments=json.dumps({"query": "Dune author"})))
    replies = [
        NS(choices=[NS(message=NS(content=None, tool_calls=[call]))]),
        NS(choices=[NS(message=NS(content="Frank Herbert.", tool_calls=None))]),
    ]

    async def create(**kw):
        return replies.pop(0)

    client = NS(chat=NS(completions=NS(create=create)))
    fails, calls, reply = await run_trial(case, client=client)
    assert fails == [] and calls == [("wikipedia_search", {"query": "Dune author"})]
