"""Structured logging, to stdout only.

One JSON object per line. The app never opens a log file: Docker captures stdout, and
docker-compose.yml caps how much of it is kept, so logs can't fill the SD card. Chat text is
not logged unless LOG_CONTENT=1, and then only short snippets.
"""
import json
import logging
import os
import sys
import time

logger = logging.getLogger("spoilerfree")
SNIPPET_CHARS = 200
_handler: logging.Handler | None = None


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "event": record.getMessage(),
            **getattr(record, "fields", {}),
        }
        if record.exc_info and record.exc_info[1]:  # type and message only: no multi-line tracebacks
            out["error"] = f"{type(record.exc_info[1]).__name__}: {record.exc_info[1]}"[:300]
        return json.dumps(out, default=str)


def emit(event: str, level: int = logging.INFO, **fields) -> None:
    """Log one event. Fields that are None are dropped."""
    logger.log(level, event, extra={"fields": {k: v for k, v in fields.items() if v is not None}})


def snippet(text: str) -> str | None:
    """A short excerpt of chat text, or None (so `emit` drops it) unless LOG_CONTENT is on."""
    if os.getenv("LOG_CONTENT", "").lower() not in ("1", "true", "yes"):
        return None
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + "..."


class _DropHealthChecks(logging.Filter):
    """Docker polls /healthz every 30s; that is thousands of useless access-log lines a day."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        return not (isinstance(args, tuple) and len(args) >= 3 and args[2] == "/healthz")


def setup_logging(env=os.environ) -> None:
    global _handler
    level = (env.get("LOG_LEVEL") or "INFO").upper()
    if level not in logging.getLevelNamesMapping():
        raise RuntimeError(f"LOG_LEVEL must be a logging level name (got {level!r})")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    if _handler:  # idempotent: the lifespan can run more than once in tests
        logger.removeHandler(_handler)
    logger.addHandler(handler)
    _handler = handler
    logger.setLevel(level)
    logger.propagate = False
    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _DropHealthChecks) for f in access.filters):
        access.addFilter(_DropHealthChecks())
