from types import SimpleNamespace as NS

import pytest

from app.chat import FALLBACK
from app.judges import JudgeResult
from bench.e2e import (E2E_QUESTIONS, JevE2EGrader, Result, TurnResult, evaluate, load_e2e_cases,
                       render_conversation, run_trial, summarize)
import asyncio

CASE = {
    "id": "c",
    "context": {"progress": "Dune, ch. 3"},
    "turns": ["Who wrote it?", "And the sequel?"],
    "needs": "Names\n  Frank Herbert.",
    "spoilers": ["Paul is the Kwisatz Haderach"],
    "notes": "be strict",
}


def test_cases_load_and_are_well_formed():
    cases = load_e2e_cases()
    assert len(cases) >= 8
    for c in cases:
        assert c["context"]["progress"] and c["turns"] and c["needs"] and c["spoilers"]


class Judge:
    def __init__(self, *verdicts):
        self.verdicts = list(verdicts)

    async def judge(self, progress, history, user_message, candidate):
        return self.verdicts.pop(0)


@pytest.mark.asyncio
async def test_run_trial_carries_approved_history_between_turns():
    seen = []

    async def generate(messages):
        seen.append(messages)
        return f"reply{len(seen)}"

    turns = await run_trial(CASE, generate, Judge(JudgeResult(False), JudgeResult(False)))
    assert [t.reply for t in turns] == ["reply1", "reply2"]
    second = seen[1]
    assert second[1:3] == [{"role": "user", "content": "Who wrote it?"},
                           {"role": "assistant", "content": "reply1"}]
    assert "Dune, ch. 3" in second[0]["content"]  # progress reaches the chat model


@pytest.mark.asyncio
async def test_run_trial_records_rejections_and_fallbacks_and_keeps_the_fallback_as_history():
    drafts = iter(["s1", "s2", "fine"])
    seen = []

    async def generate(messages):
        seen.append(messages)
        return next(drafts)

    bad = JudgeResult(True, "r")
    turns = await run_trial(CASE, generate, Judge(bad, bad, JudgeResult(False)))
    assert turns[0].reply == FALLBACK and turns[0].blocked and turns[0].spoiler_prevented
    assert turns[0].attempts == 2
    assert turns[1].reply == "fine" and not turns[1].spoiler_prevented
    assert {"role": "assistant", "content": FALLBACK} in seen[2]  # what the reader saw, not the drafts
    assert all(m["content"] not in ("s1", "s2") for m in seen[2])


def test_grader_sees_ground_truth_and_the_conversation_but_not_pipeline_internals():
    turns = [TurnResult("Who wrote it?", "Frank Herbert.", 2, True, False, 1.0),
             TurnResult("And the sequel?", "Dune Messiah.", 1, False, False, 1.0)]
    text = render_conversation(CASE, turns)
    for want in ("READER PROGRESS: Dune, ch. 3", "WHAT THE READER NEEDS: Names Frank Herbert.",
                 "- Paul is the Kwisatz Haderach", "USER: Who wrote it?", "ASSISTANT: Frank Herbert.",
                 "USER: And the sequel?", "EVALUATOR NOTES: be strict"):
        assert want in text
    assert "attempts" not in text and "rejected" not in text


class FakeSystemOne:
    def __init__(self, scores):
        self.scores, self.kwargs = scores, None

    async def system_one(self, **kw):
        self.kwargs = kw
        return NS(nouls={k: NS(noul=v) for k, v in self.scores.items()})


@pytest.mark.asyncio
async def test_grader_asks_both_questions_and_applies_the_threshold():
    fake = FakeSystemOne({"spoiler_free": 0.9, "needs_met": 0.3})
    g = JevE2EGrader(client=fake, threshold=0.5)
    scores = await g.grade(CASE, [TurnResult("q", "a", 1, False, False, 0.1)])
    assert scores == {"spoiler_free": 0.9, "needs_met": 0.3} and g.failed(scores) == ["needs_met"]
    qs = fake.kwargs["questions"]
    assert set(qs) == set(E2E_QUESTIONS) == {"spoiler_free", "needs_met"}
    assert all(q.type == "noul" and set(q.criteria) == {"true", "false"} for q in qs.values())
    assert "LATER PLOT" in fake.kwargs["state"]


@pytest.mark.asyncio
async def test_evaluate_reports_pipeline_and_grader_errors_instead_of_crashing():
    sem = asyncio.Semaphore(1)

    async def broken(messages):
        raise RuntimeError("api down")

    async def fine(messages):
        return "ok"

    r = await evaluate(CASE, broken, Judge(), JevE2EGrader(client=FakeSystemOne({})), sem)
    assert r.error.startswith("pipeline error RuntimeError") and not r.scores

    class BadGrader(JevE2EGrader):
        async def grade(self, case, turns):
            raise ValueError("no key")

    r = await evaluate(CASE, fine, Judge(JudgeResult(False), JudgeResult(False)), BadGrader(), sem)
    assert r.error.startswith("grader error ValueError") and len(r.turns) == 2


def test_summarize_rates_and_pipeline_stats():
    t = lambda blocked=False, prevented=False, attempts=1, s=2.0: TurnResult("q", "a", attempts, prevented or blocked, blocked, s)
    results = [
        Result("a", [t(), t(prevented=True, attempts=2)], {"spoiler_free": 0.9, "needs_met": 0.8}),
        Result("a", [t(blocked=True, attempts=2)], {"spoiler_free": 0.9, "needs_met": 0.1}),
        Result("b", [t()], {"spoiler_free": 0.2, "needs_met": 0.9}),
        Result("b", [], {}, error="pipeline error X"),
    ]
    s = summarize(results, 0.5)
    assert s["trials"] == 4 and s["errors"] == 1
    assert s["spoiler_free"] == pytest.approx(2 / 3) and s["needs_met"] == pytest.approx(2 / 3)
    assert s["both"] == pytest.approx(1 / 3)
    assert s["turns"] == 4 and s["turns_with_rejected_draft"] == 0.5 and s["turns_blocked"] == 0.25
    assert s["mean_attempts"] == 1.5 and s["mean_seconds_per_turn"] == 2.0
    assert summarize([], 0.5)["both"] is None
