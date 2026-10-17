"""Hit the real Wikipedia API. Skipped by default; run with `pytest -m live`."""
import httpx
import pytest
import pytest_asyncio

from app import wiki

pytestmark = [pytest.mark.live, pytest.mark.asyncio]


@pytest_asyncio.fixture
async def http():
    async with httpx.AsyncClient() as client:
        yield client


async def test_search_finds_work_without_exact_title(http):
    out = await wiki.search(http, "Anna Karenina Levin")
    assert "Anna Karenina" in out and "<" not in out  # markup stripped


async def test_intro_is_shorter_than_full_article(http):
    intro = await wiki.page(http, "Anna Karenina")
    full = await wiki.page(http, "Anna Karenina", full=True)
    assert intro.startswith("Anna Karenina") and "Tolstoy" in intro
    assert len(intro) < len(full)
    assert full.endswith("[truncated]")  # a whole article exceeds MAX_CHARS


async def test_redirect_title_resolves(http):
    out = await wiki.page(http, "JK Rowling")
    assert out.startswith("J. K. Rowling")


async def test_missing_article_reports_it(http):
    assert "No article" in await wiki.page(http, "Zzqxv nonexistent article 98765")


async def test_run_tool_end_to_end(http):
    out = await wiki.run_tool(http, "wikipedia_search", '{"query": "Dune Frank Herbert novel"}')
    assert "Dune" in out and not out.startswith("Lookup failed")
