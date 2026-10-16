"""Judges decide whether a candidate reply spoils anything past the reader's progress.

Implementations benchmarked against each other:
  * LLMJudge       - a normal chat completion with a JSON-schema response (baseline).
  * DecisionsJudge - OpenAI's Decisions API (gpt-6-luna, predicate question).
  * JevJudge       - TypeSafe's Jev via the TypeSafe SDK (noul question).
"""
import json
import os
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI
from typesafe_sdk import AsyncTypeSafeClient, Noul

from .policy import SPOILER_POLICY


@dataclass
class JudgeResult:
    violates: bool
    reason: str = ""
    score: float | None = None  # P(violation) when the judge reports one


DEFAULT_QUESTION = (
    "Does the candidate reply reveal, confirm, deny, hint at, or imply any plot detail "
    'from beyond the reader\'s progress? Reader\'s progress: "{progress}"'
)


def ask(question: str | None, progress: str) -> str:
    return (question or DEFAULT_QUESTION).replace("{progress}", progress)


class Judge(Protocol):
    async def judge(
        self,
        progress: str,
        history: list[dict],
        user_message: str,
        candidate: str,
        question: str | None = None,
    ) -> JudgeResult: ...


JUDGE_SYSTEM = f"""You are a spoiler checker. An assistant must follow this policy:

{SPOILER_POLICY}

Decide whether the assistant's candidate reply breaks the policy for this reader. Judge \
it in the context of the reader's question and the recent conversation. Flag any plot \
detail from beyond the reader's progress, however indirect. Do not flag what the policy \
allows: content up to their point, obvious facts, how the world's magic or technology \
works as explained so far, outside historical context, answers \
given from the reader's current understanding, or open-ended clarifying questions.

Respond with JSON: {{"violates": bool, "reason": "<one sentence, no spoilers repeated>"}}."""

SCHEMA = {
    "name": "spoiler_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {"violates": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["violates", "reason"],
        "additionalProperties": False,
    },
}


def render_context(history: list[dict], user_message: str, candidate: str) -> str:
    recent = history[-4:]
    lines = [f"{m['role'].upper()}: {m['content']}" for m in recent]
    lines.append(f"USER: {user_message}")
    return "Recent conversation:\n" + "\n".join(lines) + f"\n\nCANDIDATE REPLY:\n{candidate}"


def state(progress: str, history: list[dict], user_message: str, candidate: str) -> str:
    """Input for the API judges, which take no system prompt: policy, progress, context."""
    return (f"Spoiler policy:\n{SPOILER_POLICY}\n\nReader's progress: {progress}\n\n"
            f"{render_context(history, user_message, candidate)}")


class LLMJudge:
    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None):
        self.client = client or AsyncOpenAI()
        self.model = model or os.getenv("JUDGE_MODEL", "gpt-4.1-mini")

    async def judge(self, progress, history, user_message, candidate, question=None) -> JudgeResult:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": SCHEMA},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"READER'S PROGRESS: {progress}\n\nQUESTION: {ask(question, progress)}\n\n"
                        f"{render_context(history, user_message, candidate)}"
                    ),
                },
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        return JudgeResult(bool(data["violates"]), data.get("reason", ""))


class DecisionsJudge:
    """OpenAI Decisions API: one `predicate` question, P(spoiler) vs a threshold."""

    def __init__(self, client: AsyncOpenAI | None = None, threshold: float | None = None):
        self.client = client or AsyncOpenAI()
        self.threshold = threshold if threshold is not None else float(
            os.getenv("DECISIONS_THRESHOLD", "0.5")
        )

    async def judge(self, progress, history, user_message, candidate, question=None) -> JudgeResult:
        decision = await self.client.decisions.create(
            model="gpt-6-luna",
            input=state(progress, history, user_message, candidate),
            questions=[
                {"type": "predicate", "name": "violates_rule", "instructions": ask(question, progress)}
            ],
        )
        answer = decision.answers[0]
        if answer.type != "predicate":  # refusal or anything unexpected: fail closed
            return JudgeResult(True, f"checker returned {answer.type}", None)
        p = answer.probability
        return JudgeResult(
            p >= self.threshold,
            f"The reply likely reveals a plot detail past the reader's progress (checker probability {p:.2f}).",
            p,
        )


class JevJudge:
    """TypeSafe's Jev via the TypeSafe SDK. One `noul` question whose answer is
    P(spoiler); the client reads TYPESAFE_API_KEY."""

    CRITERIA = {
        "true": "The reply reveals, confirms, denies, hints at, or implies a plot detail past the reader's progress",
        "false": "The reply contains no spoilers for this reader",
    }

    def __init__(self, client: AsyncTypeSafeClient | None = None, threshold: float | None = None):
        self._client = client
        self.threshold = threshold if threshold is not None else float(
            os.getenv("JEV_THRESHOLD", "0.5")
        )

    @property
    def client(self) -> AsyncTypeSafeClient:
        if self._client is None:  # lazy: constructing it requires the API key
            self._client = AsyncTypeSafeClient(model="jev-latest")
        return self._client

    async def judge(self, progress, history, user_message, candidate, question=None) -> JudgeResult:
        response = await self.client.system_one(
            state=state(progress, history, user_message, candidate),
            questions={
                "violates_rule": Noul(instructions=ask(question, progress), criteria=self.CRITERIA)
            },
        )
        p = float(response.nouls["violates_rule"].noul)
        return JudgeResult(
            p >= self.threshold,
            f"The reply likely reveals a plot detail past the reader's progress (checker probability {p:.2f}).",
            p,
        )


def make_judge(name: str | None = None) -> Judge:
    name = name or os.getenv("JUDGE", "llm")
    if name == "llm":
        return LLMJudge()
    if name == "decisions":
        return DecisionsJudge()
    if name == "jev":
        return JevJudge()
    raise ValueError(f"unknown judge {name!r}")
