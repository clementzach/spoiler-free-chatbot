import os
from dataclasses import dataclass
from typing import Awaitable, Callable

from openai import AsyncOpenAI

from .judges import Judge, JudgeResult

MAX_ATTEMPTS = 2
FALLBACK = (
    "I couldn't answer that without risking a spoiler under your rule. "
    "Try rephrasing, or ask about something else."
)

Generate = Callable[[list[dict]], Awaitable[str]]


def system_prompt(rule: str) -> str:
    return (
        "You are a friendly assistant for discussing books, films and shows.\n"
        f"The user's spoiler rule, which you must obey in every reply:\n\"{rule}\"\n"
        "Do not reveal, confirm, deny, or hint at anything the rule protects. If a "
        "question can't be answered within the rule, say you can't answer that "
        "without breaking it, and offer what you can talk about instead."
    )


def openai_generate(client: AsyncOpenAI | None = None, model: str | None = None) -> Generate:
    client = client or AsyncOpenAI()
    model = model or os.getenv("MAIN_MODEL", "gpt-4.1")

    async def generate(messages: list[dict]) -> str:
        resp = await client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""

    return generate


@dataclass
class ChatResult:
    reply: str
    spoiler_prevented: bool  # at least one draft was rejected
    blocked: bool  # every attempt was rejected; reply is the fallback
    attempts: int
    rule: str


async def respond(
    generate: Generate,
    judge: Judge,
    rule: str,
    history: list[dict],
    user_message: str,
    max_attempts: int = MAX_ATTEMPTS,
) -> ChatResult:
    """`history` holds only approved turns. Rejected drafts live in `extra` for the
    retry request and are never returned, so they can't leak into later turns."""
    base = [{"role": "system", "content": system_prompt(rule)}, *history,
            {"role": "user", "content": user_message}]
    extra: list[dict] = []
    for attempt in range(1, max_attempts + 1):
        draft = await generate(base + extra)
        try:
            verdict = await judge.judge(rule, history, user_message, draft)
        except Exception as exc:  # fail closed
            verdict = JudgeResult(True, f"checker error: {type(exc).__name__}")
        if not verdict.violates:
            return ChatResult(draft, attempt > 1, False, attempt, rule)
        extra = [
            {"role": "assistant", "content": draft},
            {
                "role": "user",
                "content": (
                    "[Automated spoiler check] That draft was rejected because it "
                    f"broke the rule \"{rule}\". Problem: {verdict.reason} "
                    "Write a new reply that obeys the rule. Don't mention this check."
                ),
            },
        ]
    return ChatResult(FALLBACK, True, True, max_attempts, rule)
