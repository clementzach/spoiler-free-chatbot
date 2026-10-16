# Spoiler-free chatbot

Type a spoiler rule, chat normally. Each draft reply is checked by a judge; a draft
that breaks the rule is rejected and regenerated (max 2 attempts), otherwise a safe
fallback is shown. The UI shows a "Spoiler prevented" badge whenever a draft was rejected.

## Run locally
    python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
    cp .env.example .env   # fill in keys, then: set -a; . ./.env; set +a
    .venv/bin/uvicorn app.main:app --reload

## Configuration
See `.env.example`. `OPENAI_API_KEY` is always required (chat model); `TYPESAFE_API_KEY`
is also required when `JUDGE=jev`. The app refuses to start if a required key is missing.
Optional: `JUDGE` (`llm` default, `decisions`, `jev`), `MAIN_MODEL`, `JUDGE_MODEL`,
`DECISIONS_THRESHOLD`, `JEV_THRESHOLD`, `PORT`.

## Deploy (Docker)
    docker build -t spoiler-free-chatbot .
    docker run --env-file .env -p 8000:8000 spoiler-free-chatbot
    # or: docker compose up --build

The image runs as a non-root user, listens on `$PORT` (default 8000) and serves
`GET /healthz` for health checks. It works on any container host (Fly, Render, Cloud Run,
ECS...): set the env vars as secrets there. The server holds no state, so you can run
several instances. Put it behind HTTPS (the host usually does this).

Warning: there is no authentication or rate limiting, and every message costs API calls.
Don't expose a public instance without adding one or both at the proxy or app level.

No server state: the browser sends the approved history and the current rule each turn.

## Benchmarking judges
Each benchmark is one file in `bench/cases/` with three editable parts:
`context` (rule, history, user message), `response` (the chat reply to evaluate) and
`judge` (the question every judge is asked), plus the `expected` label.
Copy `bench/cases/_template.yaml` to add one.

    .venv/bin/python -m bench.run --judges llm decisions -v
    .venv/bin/python -m bench.run --only sw-hint hp-reassure

Judges: `llm` (chat-completion baseline), `decisions` (OpenAI Decisions API, predicate
question thresholded at `DECISIONS_THRESHOLD`, default 0.5), `jev` (TypeSafe's Jev via `typesafe-sdk`, `noul` question thresholded at `JEV_THRESHOLD`, default 0.5; needs `TYPESAFE_API_KEY`).
