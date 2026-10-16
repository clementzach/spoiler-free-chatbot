import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .chat import ChatResult, openai_generate, respond
from .judges import make_judge

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
    rule: str = Field(min_length=1, max_length=1000)
    message: str = Field(min_length=1, max_length=4000)
    history: list[Turn] = []  # approved turns only; the server keeps no state


class ChatResponse(BaseModel):
    reply: str
    spoiler_prevented: bool
    blocked: bool
    attempts: int
    rule: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResult:
    generate, judge = deps()
    return await respond(
        generate, judge, req.rule.strip(), [t.model_dump() for t in req.history], req.message
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
