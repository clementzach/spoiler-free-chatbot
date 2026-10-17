import logging

import pytest

from app import resilience


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Retries back off for real in production; tests record the delays instead of waiting."""
    delays = []

    async def fake_sleep(seconds):
        delays.append(seconds)

    monkeypatch.setattr(resilience, "sleep", fake_sleep)
    return delays


@pytest.fixture
def events():
    """Structured log events emitted during a test, as (event, level, fields)."""
    seen = []

    class Capture(logging.Handler):
        def emit(self, record):
            seen.append((record.getMessage(), record.levelno, getattr(record, "fields", {})))

    handler = Capture()
    from app.log import logger

    old = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield seen
    logger.removeHandler(handler)
    logger.setLevel(old)
