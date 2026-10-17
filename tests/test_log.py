import io
import json
import logging

import pytest

from app import log
from app.chat import FALLBACK, respond
from app.judges import JudgeResult


def names(events):
    return [e for e, _, _ in events]


def test_formatter_emits_one_json_object_per_event_and_drops_none():
    rec = logging.LogRecord("spoilerfree", logging.WARNING, __file__, 1, "attempt", None, None)
    rec.fields = {"attempt": 2, "ms": 5}
    out = json.loads(log.JsonFormatter().format(rec))
    assert out["event"] == "attempt" and out["level"] == "WARNING" and out["attempt"] == 2
    assert out["ts"].endswith("Z")


def test_emit_drops_none_fields(events):
    log.emit("x", a=1, b=None)
    assert events == [("x", logging.INFO, {"a": 1})]


def test_snippet_only_when_log_content_is_on(monkeypatch):
    monkeypatch.delenv("LOG_CONTENT", raising=False)
    assert log.snippet("hello") is None
    monkeypatch.setenv("LOG_CONTENT", "1")
    assert log.snippet("hello") == "hello"
    assert len(log.snippet("x" * 1000)) == log.SNIPPET_CHARS + 3


def test_setup_logging_writes_json_to_stdout_only_and_is_idempotent(monkeypatch):
    out = io.StringIO()
    monkeypatch.setattr("sys.stdout", out)
    saved = log.logger.handlers[:], log.logger.level, log.logger.propagate, log._handler
    try:
        log.setup_logging({"LOG_LEVEL": "warning"})
        log.setup_logging({"LOG_LEVEL": "warning"})
        assert len([h for h in log.logger.handlers if h is log._handler]) == 1
        assert not any(isinstance(h, logging.FileHandler) for h in log.logger.handlers)
        log.emit("quiet")  # below WARNING
        log.emit("loud", level=logging.WARNING, n=1)
        lines = out.getvalue().splitlines()
        assert [json.loads(l)["event"] for l in lines] == ["loud"]
    finally:
        log.logger.handlers[:], log.logger.level, log.logger.propagate, log._handler = saved


def test_setup_logging_rejects_bad_level():
    with pytest.raises(RuntimeError, match="LOG_LEVEL"):
        log.setup_logging({"LOG_LEVEL": "loud"})


def test_healthcheck_lines_are_dropped_from_the_access_log():
    f = log._DropHealthChecks()

    def rec(path):
        return logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1,
                                 '%s - "%s %s HTTP/%s" %d', ("1.2.3.4", "GET", path, "1.1", 200), None)

    assert not f.filter(rec("/healthz"))
    assert f.filter(rec("/chat"))


class Scripted:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    async def judge(self, progress, history, user_message, candidate):
        v = self.verdicts.pop(0)
        if isinstance(v, Exception):
            raise v
        return v


def gen(drafts):
    drafts = list(drafts)

    async def generate(messages):
        return drafts.pop(0)

    return generate


@pytest.mark.asyncio
async def test_respond_logs_each_attempt_and_one_summary_without_chat_text(events, monkeypatch):
    monkeypatch.delenv("LOG_CONTENT", raising=False)
    bad = JudgeResult(True, "reveals the ending", 0.9)
    await respond(gen(["secret draft", "safe"]), Scripted([bad, JudgeResult(False)]), "ch. 3", [], "private question")
    assert names(events) == ["attempt", "attempt", "chat_turn"]
    first, second, turn = (f for _, _, f in events)
    assert first["violates"] and first["reason"] == "reveals the ending" and first["score"] == 0.9
    assert not second["violates"] and "reason" not in second
    assert turn["attempts"] == 2 and turn["spoiler_prevented"] and not turn["blocked"]
    assert "secret draft" not in json.dumps(events) and "private question" not in json.dumps(events)


@pytest.mark.asyncio
async def test_log_content_adds_snippets(events, monkeypatch):
    monkeypatch.setenv("LOG_CONTENT", "1")
    bad = JudgeResult(True, "r")
    await respond(gen(["secret draft", "safe"]), Scripted([bad, JudgeResult(False)]), "ch. 3", [], "my question")
    assert events[0][2]["draft"] == "secret draft"
    assert events[-1][2]["message"] == "my question"


@pytest.mark.asyncio
async def test_judge_error_logs_and_returns_fallback_without_regenerating(events):
    calls = []

    async def generate(messages):
        calls.append(messages)
        return "draft"

    r = await respond(generate, Scripted([RuntimeError("down")]), "ch. 3", [], "hi")
    assert r.reply == FALLBACK and r.blocked and r.attempts == 1 and len(calls) == 1
    assert names(events) == ["judge_error", "chat_turn"]
    assert events[0][1] == logging.ERROR and events[0][2]["error"] == "RuntimeError"
    assert events[1][1] == logging.WARNING  # blocked turns are warnings
