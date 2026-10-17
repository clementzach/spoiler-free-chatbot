import pytest

from app.main import app, check_config
from fastapi.testclient import TestClient


def test_check_config_requires_keys_for_selected_judge():
    check_config({"OPENAI_API_KEY": "x"})
    check_config({"OPENAI_API_KEY": "x", "JUDGE": "decisions"})
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        check_config({})
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        check_config({"OPENAI_API_KEY": "x", "JUDGE": "jev"})
    with pytest.raises(RuntimeError, match="JUDGE"):
        check_config({"OPENAI_API_KEY": "x", "JUDGE": "bogus"})


def test_healthz_and_index(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    with TestClient(app) as c:  # runs lifespan, i.e. the config check
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/").status_code == 200


def _client(monkeypatch, generate, judge=None):
    import app.main as main

    async def ok_judge(*a, **k):
        from app.judges import JudgeResult
        return JudgeResult(False)

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(main, "deps", lambda: (generate, judge or NS_judge(ok_judge)))
    return TestClient(app)


def NS_judge(fn):
    from types import SimpleNamespace
    return SimpleNamespace(judge=fn)


BODY = {"progress": "ch. 3", "message": "hi"}


def test_chat_ok(monkeypatch):
    async def generate(messages):
        return "hello"

    with _client(monkeypatch, generate) as c:
        r = c.post("/chat", json=BODY)
        assert r.status_code == 200 and r.json()["reply"] == "hello"


def test_upstream_failure_is_a_503_with_a_readable_message(monkeypatch, events):
    import openai

    async def generate(messages):
        raise openai.OpenAIError("quota")

    with _client(monkeypatch, generate) as c:
        r = c.post("/chat", json=BODY)
    assert r.status_code == 503 and "unavailable" in r.json()["detail"]
    assert [e for e, _, _ in events if e == "chat_upstream_error"]


def test_slow_turn_is_a_504(monkeypatch, events):
    import asyncio

    import app.main as main

    async def generate(messages):
        await asyncio.sleep(5)
        return "late"

    monkeypatch.setattr(main, "CHAT_TIMEOUT", 0.05)
    with _client(monkeypatch, generate) as c:
        r = c.post("/chat", json=BODY)
    assert r.status_code == 504 and "too long" in r.json()["detail"]
    assert [e for e, _, _ in events if e == "chat_timeout"]
