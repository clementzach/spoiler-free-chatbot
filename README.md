# Spoiler-free chatbot

Say what you're reading and how far you've got, then chat normally. A fixed spoiler
policy (`app/policy.py`) governs both the chat model and the judge: nothing past your
point in the plot, but obvious facts, outside historical context and answers from your
current understanding are fine, and when unsure the bot asks what you already know.
Each draft reply is checked by a judge; a draft that spoils is rejected and regenerated
(max 2 attempts), otherwise a safe fallback is shown. The UI shows a "Spoiler prevented" badge whenever a draft was rejected.

## Run locally
    python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
    cp .env.example .env   # fill in keys, then: set -a; . ./.env; set +a
    .venv/bin/uvicorn app.main:app --reload

## Wikipedia grounding
The chat model can call two tools (`app/wiki.py`): `wikipedia_search` (full-text search, so
no exact title needed) and `wikipedia_page` (intro by default, whole article on request,
truncated). Free, no key. Articles often contain plot sections, so the judge still checks every
draft, but it doesn't see what the model looked up.

## Configuration
See `.env.example`. `OPENAI_API_KEY` is always required (chat model); `TYPESAFE_API_KEY`
is also required when `JUDGE=jev`. The app refuses to start if a required key is missing.
Optional: `JUDGE` (`llm` default, `decisions`, `jev`), `MAIN_MODEL` (default `gpt-6-luna`), `JUDGE_MODEL`,
`DECISIONS_THRESHOLD`, `JEV_THRESHOLD`, `PORT`.

## Deploy on a Raspberry Pi (Docker)
Use a 64-bit Raspberry Pi OS on a Pi 3/4/5 (the Python dependencies ship prebuilt arm64
wheels; 32-bit may try to compile them). Install Docker, then:

    git clone <this repo> && cd spoiler-free-chatbot
    cp .env.example .env    # add your keys
    docker compose up -d --build

The compose file uses `restart: unless-stopped`, so with Docker enabled at boot
(`sudo systemctl enable docker`) the app comes back after a reboot. Open
`http://<pi-address>:8000` from any device on your network. Check status with
`docker compose ps` / `docker compose logs -f`. The image runs as a non-root user and
serves `GET /healthz`. Docker isn't required: the Run locally steps work on the Pi too.

There is no authentication or rate limiting, so keep it on your home network. Don't
port-forward it to the internet without adding both, since every message costs API calls.

No server state: the browser sends the approved history and the reader's progress each turn.

Live checks against the real Wikipedia API (search, intro vs full article, redirects, missing
titles) are skipped by default so tests stay offline: `.venv/bin/pytest -m live`.

## Tool-use evals
`bench/tool_cases/` checks how the chat model uses the Wikipedia tools. The model is live
(needs `OPENAI_API_KEY`); Wikipedia is a per-case fixture, so results don't depend on the
network. Checks cover: looks up when it should and not for chit-chat, queries mention the
work (including when the title comes from the reader's progress), fetches the article when a
snippet isn't enough, stays on the intro rather than `full=true`, recovers from failed or
empty lookups, and doesn't repeat a spoiler planted in the fixture. Copy
`bench/tool_cases/_template.yaml` to add one.

    .venv/bin/python -m bench.tools --trials 3 -v
    .venv/bin/python -m bench.tools --trials 3 --jev
    .venv/bin/python -m bench.tools --only intro-only-by-default

Models vary run to run, so read the pass rate over several trials. Reply checks are
substring matches. `--jev` additionally sends each transcript (tool calls, tool results, final
reply) to Jev, which scores four questions: were lookups warranted, were the queries
relevant, is the reply grounded in what the tools returned, is it spoiler-free. A trial
passes Jev when every score is at least `JEV_THRESHOLD` (default 0.5); needs
`TYPESAFE_API_KEY`. A case's optional `notes:` is passed to the grader.

## Benchmarking judges
Each benchmark is one file in `bench/cases/` with three editable parts:
`context` (reader's progress, history, user message), `response` (the chat reply to evaluate) and
`judge` (the question every judge is asked), plus the `expected` label.
Copy `bench/cases/_template.yaml` to add one.

    .venv/bin/python -m bench.run --judges llm decisions -v
    .venv/bin/python -m bench.run --only sw-hint hp-reassure

Judges: `llm` (chat-completion baseline), `decisions` (OpenAI Decisions API, predicate
question thresholded at `DECISIONS_THRESHOLD`, default 0.5), `jev` (TypeSafe's Jev via `typesafe-sdk`, `noul` question thresholded at `JEV_THRESHOLD`, default 0.5; needs `TYPESAFE_API_KEY`).
