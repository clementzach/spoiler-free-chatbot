import pytest
from openai import AsyncOpenAI

from app.judges import DecisionsJudge
from bench.run import load_cases


def test_cases_load_and_have_three_parts():
    cases = load_cases()
    assert len(cases) >= 10
    for c in cases:
        assert c["context"]["rule"] and c["context"]["user_message"]
        assert c["response"] and "judge" in c


class Ans:
    def __init__(self, type, probability=None):
        self.type, self.probability = type, probability


class FakeDecisions:
    def __init__(self, answer):
        self.answer, self.kwargs = answer, None

    async def create(self, **kw):
        self.kwargs = kw
        return type("D", (), {"answers": [self.answer]})()


def judge_with(answer, threshold=0.5):
    j = DecisionsJudge(client=AsyncOpenAI(api_key="x"), threshold=threshold)
    j.client = type("C", (), {"decisions": FakeDecisions(answer)})()
    return j


@pytest.mark.asyncio
async def test_decisions_threshold_and_request_shape():
    j = judge_with(Ans("predicate", 0.8))
    r = await j.judge("no ending", [], "q", "reply", "Custom? {rule}")
    assert r.violates and r.score == 0.8
    kw = j.client.decisions.kwargs
    assert kw["model"] == "gpt-6-luna" and kw["questions"][0]["type"] == "predicate"
    assert kw["questions"][0]["instructions"] == "Custom? no ending"
    assert not (await judge_with(Ans("predicate", 0.2)).judge("r", [], "q", "c")).violates


@pytest.mark.asyncio
async def test_decisions_refusal_fails_closed():
    assert (await judge_with(Ans("refusal")).judge("r", [], "q", "c")).violates


class FakeSystemOne:
    def __init__(self, p):
        self.p, self.kwargs = p, None

    async def system_one(self, **kw):
        self.kwargs = kw
        return type("R", (), {"nouls": {"violates_rule": type("N", (), {"noul": self.p})()}})()


@pytest.mark.asyncio
async def test_jev_request_shape_and_threshold():
    from app.judges import JevJudge

    fake = FakeSystemOne(0.7)
    r = await JevJudge(client=fake, threshold=0.5).judge("no ending", [], "q", "reply", "Custom? {rule}")
    assert r.violates and r.score == 0.7
    q = fake.kwargs["questions"]["violates_rule"]
    assert q.type == "noul" and q.instructions == "Custom? no ending"
    assert set(q.criteria) == {"true", "false"} and "reply" in fake.kwargs["state"]
    assert not (await JevJudge(client=fake, threshold=0.9).judge("r", [], "q", "c")).violates
