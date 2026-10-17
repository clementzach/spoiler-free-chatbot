import logging
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import httpx
from openai import AsyncOpenAI

from . import wiki
from .judges import Judge
from .log import emit, snippet
from .policy import SPOILER_POLICY
from .resilience import openai_client

MAX_ATTEMPTS = 2
MAX_TOOL_ROUNDS = 4
FALLBACK = (
    "I couldn't answer that without risking a spoiler for where you are. "
    "Try rephrasing, or ask about something else."
)

Generate = Callable[[list[dict]], Awaitable[str]]


def _ms(since: float) -> int:
    return round((time.perf_counter() - since) * 1000)


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


ToolRunner = Callable[[str, str], Awaitable[str]]  # (tool name, JSON arguments) -> result text


def openai_generate(
    client: AsyncOpenAI | None = None,
    model: str | None = None,
    run_tool: ToolRunner | None = None,
) -> Generate:
    """`run_tool` defaults to live Wikipedia; benchmarks inject a stub."""
    client = client or openai_client()
    model = model or os.getenv("MAIN_MODEL", "gpt-6-luna")

    async def loop(messages: list[dict], run: ToolRunner) -> str:
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
                t0 = time.perf_counter()
                result = await run(c.function.name, c.function.arguments)
                failed = result.startswith(wiki.LOOKUP_FAILED)
                emit("tool_call", level=logging.WARNING if failed else logging.INFO,
                     tool=c.function.name, args=c.function.arguments[:120], ms=_ms(t0),
                     result_chars=len(result), error=result[:100] if failed else None)
                messages.append({"role": "tool", "tool_call_id": c.id, "content": result})
        # Out of lookups: ask for an answer without tools.
        emit("tool_rounds_exhausted", level=logging.WARNING, rounds=MAX_TOOL_ROUNDS)
        resp = await client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""

    async def generate(messages: list[dict]) -> str:
        if run_tool is not None:
            return await loop(list(messages), run_tool)
        async with httpx.AsyncClient() as http:
            return await loop(list(messages), lambda n, a: wiki.run_tool(http, n, a))

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
    started = time.perf_counter()

    def finish(reply: str, prevented: bool, blocked: bool, attempts: int) -> ChatResult:
        emit("chat_turn", level=logging.WARNING if blocked else logging.INFO,
             attempts=attempts, spoiler_prevented=prevented, blocked=blocked,
             history_messages=len(history), ms=_ms(started), progress=progress[:80],
             message=snippet(user_message))
        return ChatResult(reply, prevented, blocked, attempts, progress)

    for attempt in range(1, max_attempts + 1):
        t0 = time.perf_counter()
        draft = await generate(base + extra)
        draft_ms = _ms(t0)
        t0 = time.perf_counter()
        try:
            verdict = await judge.judge(progress, history, user_message, draft)
        except Exception as exc:  # fail closed. The judge client already retried, so don't regenerate
            emit("judge_error", level=logging.ERROR, attempt=attempt, error=type(exc).__name__,
                 detail=str(exc)[:200], draft_ms=draft_ms, judge_ms=_ms(t0))
            return finish(FALLBACK, True, True, attempt)
        emit("attempt", attempt=attempt, draft_ms=draft_ms, judge_ms=_ms(t0),
             violates=verdict.violates, score=verdict.score,
             reason=verdict.reason if verdict.violates else None,
             draft=snippet(draft) if verdict.violates else None)
        if not verdict.violates:
            return finish(draft, attempt > 1, False, attempt)
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
    return finish(FALLBACK, True, True, max_attempts)
