import pytest

from app.chat import SPOILER_CAUGHT, respond
from app.judges import InputTooLong, JudgeResult, LLMJudge


class ScriptedJudge:
    max_input_tokens = 10_000

    def __init__(self, verdicts):
        self.verdicts = list(verdicts)
        self.seen = []

    def render(self, progress, history, user_message, candidate):
        return progress + user_message + candidate + "".join(m["content"] for m in history)

    async def judge(self, progress, history, user_message, candidate):
        self.seen.append(history)
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
async def test_clean_reply():
    seen = []
    r = await respond(make_generate(["fine"], seen), ScriptedJudge([OK]), "ch. 3", [], "hi")
    assert (r.reply, r.blocked) == ("fine", False)
    assert "ch. 3" in seen[0][0]["content"]  # main model sees the progress
    assert "Never reveal plot details" in seen[0][0]["content"]  # and the fixed policy


@pytest.mark.asyncio
async def test_caught_spoiler_is_not_retried_and_never_returned():
    seen = []
    r = await respond(make_generate(["spoiler", "unused"], seen), ScriptedJudge([BAD]), "ch. 3", [], "hi")
    assert (r.reply, r.blocked) == (SPOILER_CAUGHT, True)
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_judge_error_propagates():
    with pytest.raises(RuntimeError):
        await respond(make_generate(["a"], []), ScriptedJudge([RuntimeError("boom")]), "ch. 3", [], "hi")


@pytest.mark.asyncio
async def test_judge_sees_whole_history():
    history = [{"role": "user", "content": f"q{i}"} for i in range(10)]
    judge = ScriptedJudge([OK])
    await respond(make_generate(["ok"], []), judge, "ch. 3", history, "hi")
    assert judge.seen[0] == history


@pytest.mark.asyncio
async def test_too_long_fails_before_generating():
    seen = []
    judge = ScriptedJudge([OK])
    judge.max_input_tokens = 5
    with pytest.raises(InputTooLong):
        await respond(make_generate(["x"], seen), judge, "ch. 3", [], "a long enough message")
    assert seen == []


@pytest.mark.asyncio
async def test_real_judge_render_includes_full_history_and_enforces_limit():
    j = LLMJudge(client=object(), max_input_tokens=1_000)
    history = [{"role": "user", "content": f"turn {i}"} for i in range(8)]
    assert "turn 0" in j.render("ch. 3", history, "hi", "reply")
    with pytest.raises(InputTooLong):
        await j.judge("ch. 3", [{"role": "user", "content": "x" * 5_000}], "hi", "reply")
