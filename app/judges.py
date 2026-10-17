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

from .log import emit
from .policy import SPOILER_POLICY
from .resilience import openai_client, typesafe_client


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


# Judges see the whole chat, oldest turns dropped only if it outgrows the judge's context
# window. The window is a conservative default: raise JUDGE_CONTEXT_TOKENS if yours is bigger.
DEFAULT_CONTEXT_TOKENS = 100_000
RESERVED_TOKENS = 4_000  # the judge's own output and request overhead
CHARS_PER_TOKEN = 3  # English averages ~4; 3 leaves headroom without needing a tokenizer


def context_tokens() -> int:
    return int(os.getenv("JUDGE_CONTEXT_TOKENS", DEFAULT_CONTEXT_TOKENS))


def history_room(window_tokens: int, *fixed_parts: str) -> int:
    """Characters left for the chat history once the fixed parts of the request are in."""
    return max(0, (window_tokens - RESERVED_TOKENS) * CHARS_PER_TOKEN - sum(map(len, fixed_parts)))


def _line(m: dict) -> str:
    return f"{m['role'].upper()}: {m['content']}"


def fit_history(history: list[dict], room: int | None) -> list[dict]:
    """The newest whole messages that fit in `room` characters (all of them if None)."""
    if room is None:
        return history
    kept, used = [], 0
    for m in reversed(history):
        used += len(_line(m)) + 1
        if used > room:
            break
        kept.append(m)
    return kept[::-1]


def render_context(history: list[dict], user_message: str, candidate: str, room: int | None = None) -> str:
    kept = fit_history(history, room)
    lines = [_line(m) for m in kept]
    if len(kept) < len(history):
        dropped = len(history) - len(kept)
        emit("judge_history_trimmed", dropped=dropped, kept=len(kept))
        lines.insert(0, f"[{dropped} earlier messages omitted: the chat is too long for the checker]")
    lines.append(f"USER: {user_message}")
    return "Conversation so far:\n" + "\n".join(lines) + f"\n\nCANDIDATE REPLY:\n{candidate}"


def state(progress: str, history: list[dict], user_message: str, candidate: str,
          window_tokens: int | None = None) -> str:
    """Input for the API judges, which take no system prompt: policy, progress, context."""
    room = history_room(window_tokens or context_tokens(), SPOILER_POLICY, progress, user_message, candidate)
    return (f"Spoiler policy:\n{SPOILER_POLICY}\n\nReader's progress: {progress}\n\n"
            f"{render_context(history, user_message, candidate, room)}")


class LLMJudge:
    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None,
                 window_tokens: int | None = None):
        self.client = client or openai_client()
        self.model = model or os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
        self.window_tokens = window_tokens or context_tokens()

    async def judge(self, progress, history, user_message, candidate, question=None) -> JudgeResult:
        asked = ask(question, progress)
        room = history_room(self.window_tokens, JUDGE_SYSTEM, progress, asked, user_message, candidate)
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": SCHEMA},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"READER'S PROGRESS: {progress}\n\nQUESTION: {asked}\n\n"
                        f"{render_context(history, user_message, candidate, room)}"
                    ),
                },
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        return JudgeResult(bool(data["violates"]), data.get("reason", ""))


class DecisionsJudge:
    """OpenAI Decisions API: one `predicate` question, P(spoiler) vs a threshold."""

    def __init__(self, client: AsyncOpenAI | None = None, threshold: float | None = None,
                 window_tokens: int | None = None):
        self.client = client or openai_client()
        self.threshold = threshold if threshold is not None else float(
            os.getenv("DECISIONS_THRESHOLD", "0.5")
        )
        self.window_tokens = window_tokens or context_tokens()

    async def judge(self, progress, history, user_message, candidate, question=None) -> JudgeResult:
        decision = await self.client.decisions.create(
            model="gpt-6-luna",
            input=state(progress, history, user_message, candidate, self.window_tokens),
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

    def __init__(self, client: AsyncTypeSafeClient | None = None, threshold: float | None = None,
                 window_tokens: int | None = None):
        self._client = client
        self.threshold = threshold if threshold is not None else float(
            os.getenv("JEV_THRESHOLD", "0.5")
        )
        self.window_tokens = window_tokens or context_tokens()

    @property
    def client(self) -> AsyncTypeSafeClient:
        if self._client is None:  # lazy: constructing it requires the API key
            self._client = typesafe_client()
        return self._client

    async def judge(self, progress, history, user_message, candidate, question=None) -> JudgeResult:
        response = await self.client.system_one(
            state=state(progress, history, user_message, candidate, self.window_tokens),
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
