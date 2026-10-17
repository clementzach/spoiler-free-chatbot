"""Timeouts and retries for the services the app depends on.

OpenAI and TypeSafe clients retry internally (backoff, Retry-After), so they are only
configured here. Wikipedia goes through plain httpx, so `with_retries` does it by hand.
"""
import logging
import os
import random
from asyncio import sleep
from typing import Awaitable, Callable, TypeVar

import httpx
from openai import AsyncOpenAI
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from .log import emit

T = TypeVar("T")
TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
MAX_BACKOFF = 5.0


def api_timeout() -> float:
    return float(os.getenv("API_TIMEOUT", "60"))  # seconds, per attempt


def api_retries() -> int:
    return int(os.getenv("API_MAX_RETRIES", "3"))  # after the first attempt


def openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(timeout=api_timeout(), max_retries=api_retries())


def typesafe_client() -> AsyncTypeSafeClient:
    return AsyncTypeSafeClient(
        model="jev-latest", timeout=api_timeout(), retry=RetryPolicy(max_retries=api_retries())
    )


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in TRANSIENT_STATUS
    return isinstance(exc, httpx.TransportError)  # connection errors and timeouts


def _retry_after(exc: BaseException) -> float | None:
    if isinstance(exc, httpx.HTTPStatusError):
        try:
            return float(exc.response.headers["Retry-After"])
        except (KeyError, ValueError):
            pass
    return None


async def with_retries(
    call: Callable[[], Awaitable[T]], retries: int, base_delay: float = 0.5, label: str = "call"
) -> T:
    """Run `call`, retrying transient httpx failures with jittered exponential backoff.
    Anything else (4xx, bad JSON, bugs) is raised at once."""
    for attempt in range(retries + 1):
        try:
            return await call()
        except Exception as exc:
            if attempt == retries or not is_transient(exc):
                raise
            delay = min(MAX_BACKOFF, _retry_after(exc) or base_delay * 2**attempt)
            delay *= 1 - random.random() * 0.25
            emit("retry", level=logging.WARNING, call=label, attempt=attempt + 1, error=type(exc).__name__,
                 delay_s=round(delay, 2))
            await sleep(delay)
    raise AssertionError("unreachable")
