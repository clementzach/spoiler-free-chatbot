"""Judges decide whether a candidate reply violates the user's spoiler rule.

Two implementations are meant to be benchmarked against each other:
  * LLMJudge       - a normal chat completion with a JSON-schema response.
  * DecisionsJudge - OpenAI's Decisions API (not implemented yet, see below).
"""
import json
import os
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI


@dataclass
class JudgeResult:
    violates: bool
    reason: str = ""


class Judge(Protocol):
    async def judge(
        self, rule: str, history: list[dict], user_message: str, candidate: str
    ) -> JudgeResult: ...


JUDGE_SYSTEM = """You are a strict spoiler checker. The user gave a rule about what \
they do NOT want to learn. Decide whether the assistant's candidate reply violates it.

Count as a violation: stating the protected information, confirming or denying it, \
hinting at it, or implying it (e.g. "don't worry, he's fine", "I can't say whether \
she dies" when that itself reveals something). Judge the reply in the context of the \
user's question and the recent conversation. Do not flag harmless information the rule \
does not cover. Information about other books/films/parts that the rule does not \
mention is not a violation.

Respond with JSON: {"violates": bool, "reason": "<one sentence, no spoilers repeated>"}."""

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


class LLMJudge:
    def __init__(self, client: AsyncOpenAI | None = None, model: str | None = None):
        self.client = client or AsyncOpenAI()
        self.model = model or os.getenv("JUDGE_MODEL", "gpt-4.1-mini")

    async def judge(self, rule, history, user_message, candidate) -> JudgeResult:
        resp = await self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_schema", "json_schema": SCHEMA},
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {
                    "role": "user",
                    "content": f"RULE: {rule}\n\n{render_context(history, user_message, candidate)}",
                },
            ],
        )
        data = json.loads(resp.choices[0].message.content)
        return JudgeResult(bool(data["violates"]), data.get("reason", ""))


class DecisionsJudge:
    """TODO: implement against https://developers.openai.com/api/docs/guides/decisions.

    The docs were unreachable when this was scaffolded, so the request shape is not
    guessed here. Define two answers (e.g. "ok" / "spoiler"), pass the same inputs
    LLMJudge sees (rule, recent conversation, candidate), and map the chosen answer
    to JudgeResult(violates=...). If the API returns no explanation, use a generic
    reason; the retry prompt still works without one.
    """

    async def judge(self, rule, history, user_message, candidate) -> JudgeResult:
        raise NotImplementedError("DecisionsJudge is not implemented yet")


def make_judge(name: str | None = None) -> Judge:
    name = name or os.getenv("JUDGE", "llm")
    if name == "llm":
        return LLMJudge()
    if name == "decisions":
        return DecisionsJudge()
    raise ValueError(f"unknown judge {name!r}")
