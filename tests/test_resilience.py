import httpx
import pytest

from app import resilience, wiki
from app.resilience import with_retries


def status_error(code, headers=None):
    req = httpx.Request("GET", "http://x")
    return httpx.HTTPStatusError("boom", request=req, response=httpx.Response(code, headers=headers, request=req))


def flaky(*failures):
    failures, calls = list(failures), []

    async def call():
        calls.append(1)
        if failures:
            raise failures.pop(0)
        return "ok"

    return call, calls


@pytest.mark.asyncio
async def test_retries_transient_errors_with_growing_backoff(no_backoff, events):
    call, calls = flaky(status_error(503), httpx.ConnectTimeout("slow"))
    assert await with_retries(call, retries=2) == "ok" and len(calls) == 3
    assert len(no_backoff) == 2 and 0 < no_backoff[0] <= 0.5 and 0.5 < no_backoff[1] <= 1.0
    assert [e for e, _, _ in events] == ["retry", "retry"]


@pytest.mark.asyncio
async def test_gives_up_after_the_retry_budget(no_backoff):
    call, calls = flaky(*[status_error(500)] * 5)
    with pytest.raises(httpx.HTTPStatusError):
        await with_retries(call, retries=2)
    assert len(calls) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [status_error(404), status_error(400), ValueError("bad json")])
async def test_does_not_retry_permanent_errors(exc, no_backoff):
    call, calls = flaky(exc)
    with pytest.raises(type(exc)):
        await with_retries(call, retries=3)
    assert len(calls) == 1 and no_backoff == []


@pytest.mark.asyncio
async def test_honours_retry_after_up_to_the_cap(no_backoff):
    call, _ = flaky(status_error(429, {"Retry-After": "3"}))
    await with_retries(call, retries=1)
    assert 2.25 <= no_backoff[0] <= 3
    call, _ = flaky(status_error(429, {"Retry-After": "120"}))
    await with_retries(call, retries=1)
    assert no_backoff[1] <= resilience.MAX_BACKOFF


@pytest.mark.asyncio
async def test_wikipedia_recovers_from_a_blip():
    hits = []

    def handler(req):
        hits.append(1)
        if len(hits) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"query": {"search": [{"title": "Dune", "snippet": "a novel"}]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert await wiki.search(http, "dune") == "Dune: a novel"
    assert len(hits) == 2


@pytest.mark.asyncio
async def test_wikipedia_outage_still_returns_text_to_the_model(no_backoff):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as http:
        out = await wiki.run_tool(http, "wikipedia_search", '{"query": "x"}')
    assert out == "Lookup failed: HTTPStatusError" and len(no_backoff) == wiki.RETRIES


def test_openai_client_gets_a_timeout_and_retry_budget(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.delenv("API_TIMEOUT", raising=False)
    monkeypatch.delenv("API_MAX_RETRIES", raising=False)
    c = resilience.openai_client()
    assert c.max_retries == 3 and c.timeout == 60
    monkeypatch.setenv("API_TIMEOUT", "5")
    monkeypatch.setenv("API_MAX_RETRIES", "1")
    c = resilience.openai_client()
    assert c.max_retries == 1 and c.timeout == 5
