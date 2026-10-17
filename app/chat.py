import os
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
from openai import AsyncOpenAI

from . import wiki
from .judges import Judge, JudgeResult
from .policy import SPOILER_POLICY

MAX_ATTEMPTS = 2
MAX_TOOL_ROUNDS = 4
FALLBACK = (
    "I couldn't answer that without risking a spoiler for where you are. "
    "Try rephrasing, or ask about something else."
)

Generate = Callable[[list[dict]], Awaitable[str]]


def system_prompt(progress: str) -> str:
    return (
        "You are a friendly assistant for discussing books, films and shows.\n\n"
        f"{SPOILER_POLICY}\n\n"
        "If a question can't be answered without a spoiler, say so briefly and offer "
        "what you can talk about instead.\n\n"
        "You can look things up on Wikipedia (search first; you don't need the exact title) "
        "instead of relying on memory. Articles often contain spoilers: use them only to "
        "verify what you're allowed to say, and apply the policy above to anything you "
        "quote or paraphrase from them.\n\n"
        f"What the reader is reading and how far they've got:\n\"{progress}\""
    )


def openai_generate(client: AsyncOpenAI | None = None, model: str | None = None) -> Generate:
    client = client or AsyncOpenAI()
    model = model or os.getenv("MAIN_MODEL", "gpt-6-luna")

    async def generate(messages: list[dict]) -> str:
        messages = list(messages)
        async with httpx.AsyncClient() as http:
            for _ in range(MAX_TOOL_ROUNDS):
                resp = await client.chat.completions.create(
                    model=model, messages=messages, tools=wiki.TOOLS
                )
                msg = resp.choices[0].message
                if not msg.tool_calls:
                    return msg.content or ""
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {"id": c.id, "type": "function",
                         "function": {"name": c.function.name, "arguments": c.function.arguments}}
                        for c in msg.tool_calls
                    ],
                })
                for c in msg.tool_calls:
                    result = await wiki.run_tool(http, c.function.name, c.function.arguments)
                    messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
        # Out of lookups: ask for an answer without tools.
        resp = await client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""

    return generate


@dataclass
class ChatResult:
    reply: str
    spoiler_prevented: bool  # at least one draft was rejected
    blocked: bool  # every attempt was rejected; reply is the fallback
    attempts: int
    progress: str


async def respond(
    generate: Generate,
    judge: Judge,
    progress: str,
    history: list[dict],
    user_message: str,
    max_attempts: int = MAX_ATTEMPTS,
) -> ChatResult:
    """`history` holds only approved turns. Rejected drafts live in `extra` for the
    retry request and are never returned, so they can't leak into later turns."""
    base = [{"role": "system", "content": system_prompt(progress)}, *history,
            {"role": "user", "content": user_message}]
    extra: list[dict] = []
    for attempt in range(1, max_attempts + 1):
        draft = await generate(base + extra)
        try:
            verdict = await judge.judge(progress, history, user_message, draft)
        except Exception as exc:  # fail closed
            verdict = JudgeResult(True, f"checker error: {type(exc).__name__}")
        if not verdict.violates:
            return ChatResult(draft, attempt > 1, False, attempt, progress)
        extra = [
            {"role": "assistant", "content": draft},
            {
                "role": "user",
                "content": (
                    "[Automated spoiler check] That draft was rejected as a spoiler for "
                    f"a reader at \"{progress}\". Problem: {verdict.reason} "
                    "Write a new reply with no spoilers. Don't mention this check."
                ),
            },
        ]
    return ChatResult(FALLBACK, True, True, max_attempts, progress)
