from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .chat import ChatResult, openai_generate, respond
from .judges import make_judge

app = FastAPI(title="Spoiler-free chatbot")
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


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")
