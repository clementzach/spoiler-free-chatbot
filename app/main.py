import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .chat import openai_generate, respond
from .judges import JUDGES, InputTooLong, make_judge

log = logging.getLogger(__name__)


def check_config(env=os.environ) -> None:
    """Fail at startup, not on the first chat, if required keys are missing."""
    judge = env.get("JUDGE", "llm")
    if judge not in JUDGES:
        raise RuntimeError(f"JUDGE must be one of {', '.join(JUDGES)} (got {judge!r})")
    required = {"OPENAI_API_KEY", *JUDGES[judge].required_env}  # chat model always needs OpenAI
    missing = sorted(k for k in required if not env.get(k))
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


@asynccontextmanager
async def lifespan(_: FastAPI):
    check_config()
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
    blocked: bool
    progress: str


@app.post("/chat", response_model=ChatResponse,
          responses={413: {"description": "Conversation too long"}, 503: {"description": "Upstream failure"}})
async def chat(req: ChatRequest):
    generate, judge = deps()
    try:
        return await respond(
            generate, judge, req.progress.strip(), [t.model_dump() for t in req.history], req.message
        )
    except InputTooLong as exc:
        log.warning("conversation too long: %s", exc)
        return JSONResponse({"error": "too_long"}, status_code=413)
    except Exception:  # model or checker failure; never fall back to an unchecked draft
        log.exception("chat failed")
        return JSONResponse({"error": "internal"}, status_code=503)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
