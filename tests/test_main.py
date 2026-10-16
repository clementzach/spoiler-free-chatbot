import pytest

import app.main as main
from app.judges import InputTooLong
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


class Boom:
    max_input_tokens = 10_000

    def __init__(self, exc):
        self.exc = exc

    def render(self, *a):
        return ""

    async def judge(self, *a):
        raise self.exc


async def draft(messages):
    return "draft"


@pytest.mark.parametrize("exc,status", [(RuntimeError("down"), 503), (InputTooLong("big"), 413)])
def test_chat_errors_never_return_the_draft(monkeypatch, exc, status):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr(main, "deps", lambda: (draft, Boom(exc)))
    with TestClient(app) as c:
        r = c.post("/chat", json={"progress": "ch. 3", "message": "hi"})
    assert r.status_code == status and "draft" not in r.text


def test_clients_use_configured_timeout_and_retries(monkeypatch):
    from app.judges import JevJudge, LLMJudge

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("TYPESAFE_API_KEY", "x")
    monkeypatch.setenv("REQUEST_TIMEOUT", "7")
    monkeypatch.setenv("MAX_RETRIES", "0")
    c = LLMJudge().client
    assert (c.timeout, c.max_retries) == (7.0, 0)
    assert JevJudge().client is not None  # builds with the custom RetryPolicy
