import httpx
import pytest

from app import wiki


def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_search_strips_markup():
    def handler(req):
        assert req.url.params["srsearch"] == "dune book"
        return httpx.Response(200, json={"query": {"search": [
            {"title": "Dune (novel)", "snippet": 'a <span class="x">1965</span> novel &amp; more'}]}})

    async with client(handler) as http:
        assert await wiki.search(http, "dune book") == "Dune (novel): a 1965 novel & more"


@pytest.mark.asyncio
async def test_page_intro_only_by_default_and_missing():
    seen = []

    def handler(req):
        seen.append(dict(req.url.params))
        if req.url.params["titles"] == "Nope":
            return httpx.Response(200, json={"query": {"pages": [{"title": "Nope", "missing": True}]}})
        return httpx.Response(200, json={"query": {"pages": [{"title": "Dune", "extract": "x" * 7000}]}})

    async with client(handler) as http:
        text = await wiki.page(http, "Dune")
        assert text.endswith("[truncated]") and "exintro" in seen[0]
        await wiki.page(http, "Dune", full=True)
        assert "exintro" not in seen[1]
        assert "No article" in await wiki.page(http, "Nope")


@pytest.mark.asyncio
async def test_run_tool_swallows_errors():
    def handler(req):
        return httpx.Response(500)

    async with client(handler) as http:
        assert (await wiki.run_tool(http, "wikipedia_search", '{"query": "x"}')).startswith("Lookup failed")
        assert "Unknown tool" in await wiki.run_tool(http, "nope", "{}")
