"""Judges decide whether a candidate reply spoils anything past the reader's progress.

Implementations benchmarked against each other:
  * LLMJudge       - a normal chat completion with a JSON-schema response (baseline).
  * DecisionsJudge - OpenAI's Decisions API (gpt-6-luna, predicate question).
  * JevJudge       - TypeSafe's Jev via the TypeSafe SDK (noul question).

Every judge sees the whole conversation. If that won't fit in the judge model's input
limit, it raises InputTooLong instead of judging a truncated conversation.
"""
import json
import os
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI, BadRequestError
from typesafe_sdk import AsyncTypeSafeClient, Noul, RetryPolicy

from .policy import SPOILER_POLICY


@dataclass
class JudgeResult:
    violates: bool
    reason: str = ""
    score: float | None = None  # P(violation) when the judge reports one


class InputTooLong(Exception):
    """The conversation is too long for a model to read in one request."""


def question(progress: str) -> str:
    return (
        "Does the candidate reply reveal, confirm, deny, hint at, or imply any plot detail "
        f'from beyond the reader\'s progress? Reader\'s progress: "{progress}"'
    )


def estimate_tokens(text: str) -> int:
    """Deliberately high (English averages ~4 chars/token) so we refuse a little early
    rather than send a request the model will reject or silently truncate."""
    return len(text) // 3 + 1


def is_context_error(exc: Exception) -> bool:
    return isinstance(exc, BadRequestError) and getattr(exc, "code", None) == "context_length_exceeded"


class Judge(Protocol):
    max_input_tokens: int

    def render(self, progress: str, history: list[dict], user_message: str, candidate: str) -> str:
        """Everything the judge model reads, for the input-length check."""
        ...

    async def judge(
        self, progress: str, history: list[dict], user_message: str, candidate: str
    ) -> JudgeResult: ...


def ensure_fits(judge: Judge, progress: str, history: list[dict], user_message: str,
                candidate: str = "") -> None:
    tokens = estimate_tokens(judge.render(progress, history, user_message, candidate))
    if tokens > judge.max_input_tokens:
        raise InputTooLong(
            f"~{tokens} tokens exceeds the judge's limit of {judge.max_input_tokens}"
        )


def request_timeout() -> float:
    """Seconds per attempt for every model call (chat model and judges)."""
    return float(os.getenv("REQUEST_TIMEOUT", "30"))


def max_retries() -> int:
    return int(os.getenv("MAX_RETRIES", "1"))


def openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(timeout=request_timeout(), max_retries=max_retries())


def max_tokens_env(default: int) -> int:
    return int(os.getenv("JUDGE_MAX_INPUT_TOKENS", default))


JUDGE_SYSTEM = f"""You are a spoiler checker. An assistant must follow this policy:

{SPOILER_POLICY}

Decide whether the assistant's candidate reply breaks the policy for this reader. Judge \
it in the context of the reader's question and the conversation. Flag any plot \
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
    lines = [f"{m['role'].upper()}: {m['content']}" for m in history]
    lines.append(f"USER: {user_message}")
    return "Conversation:\n" + "\n".join(lines) + f"\n\nCANDIDATE REPLY:\n{candidate}"


def state(progress: str, history: list[dict], user_message: str, candidate: str) -> str:
    """Input for the API judges, which take no system prompt: policy, progress, context."""
    return (f"Spoiler policy:\n{SPOILER_POLICY}\n\nReader's progress: {progress}\n\n"
            f"{render_context(history, user_message, candidate)}")


class LLMJudge:
    required_env = ("OPENAI_API_KEY",)

    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None,
                 max_input_tokens: int | None = None):
        self.client = client or openai_client()
        self.model = model or os.getenv("JUDGE_MODEL", "gpt-4.1-mini")
        # gpt-4.1-mini's context window; set JUDGE_MAX_INPUT_TOKENS if JUDGE_MODEL is smaller.
        self.max_input_tokens = max_input_tokens or max_tokens_env(1_047_576)

    def _user_content(self, progress, history, user_message, candidate) -> str:
        return (f"READER'S PROGRESS: {progress}\n\nQUESTION: {question(progress)}\n\n"
                f"{render_context(history, user_message, candidate)}")

    def render(self, progress, history, user_message, candidate) -> str:
        return JUDGE_SYSTEM + self._user_content(progress, history, user_message, candidate)

    async def judge(self, progress, history, user_message, candidate) -> JudgeResult:
        ensure_fits(self, progress, history, user_message, candidate)
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_schema", "json_schema": SCHEMA},
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user",
                     "content": self._user_content(progress, history, user_message, candidate)},
                ],
            )
        except BadRequestError as exc:
            if is_context_error(exc):
                raise InputTooLong(str(exc)) from exc
            raise
        data = json.loads(resp.choices[0].message.content)
        return JudgeResult(bool(data["violates"]), data.get("reason", ""))


class DecisionsJudge:
    """OpenAI Decisions API: one `predicate` question, P(spoiler) vs a threshold."""

    required_env = ("OPENAI_API_KEY",)

    def __init__(self, client: AsyncOpenAI | None = None, threshold: float | None = None,
                 max_input_tokens: int | None = None):
        self.client = client or openai_client()
        self.threshold = threshold if threshold is not None else float(
            os.getenv("DECISIONS_THRESHOLD", "0.5")
        )
        # gpt-6-luna's documented window; OpenAI hasn't published a Decisions-specific limit.
        self.max_input_tokens = max_input_tokens or max_tokens_env(1_050_000)

    def render(self, progress, history, user_message, candidate) -> str:
        return state(progress, history, user_message, candidate) + question(progress)

    async def judge(self, progress, history, user_message, candidate) -> JudgeResult:
        ensure_fits(self, progress, history, user_message, candidate)
        try:
            decision = await self.client.decisions.create(
                model="gpt-6-luna",
                input=state(progress, history, user_message, candidate),
                questions=[
                    {"type": "predicate", "name": "violates_rule", "instructions": question(progress)}
                ],
            )
        except BadRequestError as exc:
            if is_context_error(exc):
                raise InputTooLong(str(exc)) from exc
            raise
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

    required_env = ("TYPESAFE_API_KEY",)

    CRITERIA = {
        "true": "The reply reveals, confirms, denies, hints at, or implies a plot detail past the reader's progress",
        "false": "The reply contains no spoilers for this reader",
    }

    def __init__(self, client: AsyncTypeSafeClient | None = None, threshold: float | None = None,
                 max_input_tokens: int | None = None):
        self._client = client
        self.threshold = threshold if threshold is not None else float(
            os.getenv("JEV_THRESHOLD", "0.5")
        )
        # Jev 1.13's budget for state plus the longest question (sources disagree; this is
        # the most conservative documented figure).
        self.max_input_tokens = max_input_tokens or max_tokens_env(32_000)

    @property
    def client(self) -> AsyncTypeSafeClient:
        if self._client is None:  # lazy: constructing it requires the API key
            timeout, retries = request_timeout(), max_retries()
            self._client = AsyncTypeSafeClient(
                model="jev-latest",
                timeout=timeout,
                # The SDK's retry budget is per call, across all attempts.
                retry=RetryPolicy(max_retries=retries, timeout=timeout * (retries + 1)),
            )
        return self._client

    def render(self, progress, history, user_message, candidate) -> str:
        return (state(progress, history, user_message, candidate) + question(progress)
                + "".join(self.CRITERIA.values()))

    async def judge(self, progress, history, user_message, candidate) -> JudgeResult:
        ensure_fits(self, progress, history, user_message, candidate)
        response = await self.client.system_one(
            state=state(progress, history, user_message, candidate),
            questions={
                "violates_rule": Noul(instructions=question(progress), criteria=self.CRITERIA)
            },
        )
        p = float(response.nouls["violates_rule"].noul)
        return JudgeResult(
            p >= self.threshold,
            f"The reply likely reveals a plot detail past the reader's progress (checker probability {p:.2f}).",
            p,
        )


JUDGES: dict[str, type] = {"llm": LLMJudge, "decisions": DecisionsJudge, "jev": JevJudge}


def make_judge(name: str | None = None) -> Judge:
    name = name or os.getenv("JUDGE", "llm")
    if name not in JUDGES:
        raise ValueError(f"unknown judge {name!r}; choose from {', '.join(JUDGES)}")
    return JUDGES[name]()
