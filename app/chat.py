import os
from dataclasses import dataclass
from typing import Awaitable, Callable

from openai import AsyncOpenAI, BadRequestError

from .judges import InputTooLong, Judge, ensure_fits, is_context_error, openai_client
from .policy import SPOILER_POLICY

SPOILER_CAUGHT = (
    "I couldn't answer that without risking a spoiler for where you are. "
    "Try asking a different question."
)

Generate = Callable[[list[dict]], Awaitable[str]]


def system_prompt(progress: str) -> str:
    return (
        "You are a friendly assistant for discussing books, films and shows.\n\n"
        f"{SPOILER_POLICY}\n\n"
        "If a question can't be answered without a spoiler, say so briefly and offer "
        "what you can talk about instead.\n\n"
        f"What the reader is reading and how far they've got:\n\"{progress}\""
    )


def openai_generate(client: AsyncOpenAI | None = None, model: str | None = None) -> Generate:
    client = client or openai_client()
    model = model or os.getenv("MAIN_MODEL", "gpt-4.1")

    async def generate(messages: list[dict]) -> str:
        try:
            resp = await client.chat.completions.create(model=model, messages=messages)
        except BadRequestError as exc:
            if is_context_error(exc):
                raise InputTooLong(str(exc)) from exc
            raise
        return resp.choices[0].message.content or ""

    return generate


@dataclass
class ChatResult:
    reply: str
    blocked: bool  # the judge caught a spoiler; reply is SPOILER_CAUGHT, not the draft
    progress: str


async def respond(
    generate: Generate,
    judge: Judge,
    progress: str,
    history: list[dict],
    user_message: str,
) -> ChatResult:
    """One draft, one verdict. A caught spoiler is never retried: the reader gets
    SPOILER_CAUGHT and has to ask something else. Errors (including InputTooLong)
    propagate, so a draft is only ever returned after a clean verdict."""
    ensure_fits(judge, progress, history, user_message)  # before paying for a draft
    messages = [{"role": "system", "content": system_prompt(progress)}, *history,
                {"role": "user", "content": user_message}]
    draft = await generate(messages)
    verdict = await judge.judge(progress, history, user_message, draft)
    if verdict.violates:
        return ChatResult(SPOILER_CAUGHT, True, progress)
    return ChatResult(draft, False, progress)
