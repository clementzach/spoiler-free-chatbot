import pytest

from app.chat import FALLBACK, respond
from app.judges import JudgeResult


class ScriptedJudge:
    def __init__(self, verdicts):
        self.verdicts = list(verdicts)

    async def judge(self, progress, history, user_message, candidate):
        v = self.verdicts.pop(0)
        if isinstance(v, Exception):
            raise v
        return v


def make_generate(drafts, seen):
    drafts = list(drafts)

    async def generate(messages):
        seen.append(messages)
        return drafts.pop(0)

    return generate


OK, BAD = JudgeResult(False), JudgeResult(True, "reveals the ending")


@pytest.mark.asyncio
async def test_clean_first_try():
    seen = []
    r = await respond(make_generate(["fine"], seen), ScriptedJudge([OK]), "ch. 3", [], "hi")
    assert (r.reply, r.spoiler_prevented, r.blocked, r.attempts) == ("fine", False, False, 1)


@pytest.mark.asyncio
async def test_retry_then_ok_gets_feedback_and_marks_prevented():
    seen = []
    r = await respond(make_generate(["spoiler", "safe"], seen), ScriptedJudge([BAD, OK]), "ch. 3", [], "hi")
    assert (r.reply, r.spoiler_prevented, r.blocked, r.attempts) == ("safe", True, False, 2)
    retry = seen[1]
    assert retry[-2] == {"role": "assistant", "content": "spoiler"}
    assert "reveals the ending" in retry[-1]["content"]
    assert "ch. 3" in retry[0]["content"]  # main model sees the progress
    assert "Never reveal plot details" in retry[0]["content"]  # and the fixed policy


@pytest.mark.asyncio
async def test_two_failures_return_fallback_and_never_the_draft():
    r = await respond(make_generate(["s1", "s2"], []), ScriptedJudge([BAD, BAD]), "ch. 3", [], "hi")
    assert r.reply == FALLBACK and r.blocked and r.spoiler_prevented and r.attempts == 2


@pytest.mark.asyncio
async def test_judge_error_fails_closed():
    r = await respond(make_generate(["a", "b"], []), ScriptedJudge([RuntimeError("boom"), RuntimeError("boom")]), "ch. 3", [], "hi")
    assert r.blocked and r.reply == FALLBACK


@pytest.mark.asyncio
async def test_rejected_drafts_not_in_next_turn_context():
    seen = []
    r = await respond(make_generate(["spoiler", "safe"], seen), ScriptedJudge([BAD, OK]), "ch. 3", [], "hi")
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": r.reply}]
    seen2 = []
    await respond(make_generate(["next"], seen2), ScriptedJudge([OK]), "ch. 3", history, "more")
    assert all("spoiler" != m["content"] for m in seen2[0])
