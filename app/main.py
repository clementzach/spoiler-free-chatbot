import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import openai
import typesafe_sdk
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .chat import ChatResult, openai_generate, respond
from .judges import make_judge
from .log import emit, setup_logging

# Whole-turn budget. Each API call has its own timeout and retries; this bounds the sum of
# generate -> judge -> regenerate, so the browser never waits indefinitely.
CHAT_TIMEOUT = float(os.getenv("CHAT_TIMEOUT", "180"))
UPSTREAM_ERRORS = (openai.OpenAIError, typesafe_sdk.TypeSafeError)


def check_config(env=os.environ) -> None:
    """Fail at startup, not on the first chat, if required keys are missing."""
    judge = env.get("JUDGE", "llm")
    if judge not in ("llm", "decisions", "jev"):
        raise RuntimeError(f"JUDGE must be llm, decisions or jev (got {judge!r})")
    required = ["OPENAI_API_KEY"] + (["TYPESAFE_API_KEY"] if judge == "jev" else [])
    missing = [k for k in required if not env.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    check_config()
    emit("startup", judge=os.getenv("JUDGE", "llm"), model=os.getenv("MAIN_MODEL", "gpt-6-luna"),
         chat_timeout_s=CHAT_TIMEOUT)
    yield


app = FastAPI(title="Spoiler-free chatbot", lifespan=lifespan)
STATIC = Path(__file__).resolve().parent.parent / "static"

_generate = None
_judge = None


def deps():
    """Built lazily so importing the app doesn't require an API key."""
    global _generate, _judge
    if _generate is None:
        _generate, _judge = openai_generate(), make_judge()
    return _generate, _judge


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    progress: str = Field(min_length=1, max_length=1000)  # what they're reading, how far
    message: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = []  # approved turns only; the server keeps no state


class ChatResponse(BaseModel):
    reply: str
    spoiler_prevented: bool
    blocked: bool
    attempts: int
    progress: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResult:
    generate, judge = deps()
    try:
        async with asyncio.timeout(CHAT_TIMEOUT):
            return await respond(
                generate, judge, req.progress.strip(), [t.model_dump() for t in req.history],
                req.message,
            )
    except TimeoutError:
        emit("chat_timeout", level=logging.ERROR, timeout_s=CHAT_TIMEOUT)
        raise HTTPException(504, "That took too long. Please try again.")
    except UPSTREAM_ERRORS as exc:
        emit("chat_upstream_error", level=logging.ERROR, error=type(exc).__name__, detail=str(exc)[:200])
        raise HTTPException(503, "The AI service is unavailable right now. Please try again in a moment.")


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
