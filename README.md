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

## Configuration
See `.env.example`. `OPENAI_API_KEY` is always required (chat model); `TYPESAFE_API_KEY`
is also required when `JUDGE=jev`. The app refuses to start if a required key is missing.
Optional: `JUDGE` (`llm` default, `decisions`, `jev`), `MAIN_MODEL`, `JUDGE_MODEL`,
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

## Benchmarking judges
Each benchmark is one file in `bench/cases/` with three editable parts:
`context` (reader's progress, history, user message), `response` (the chat reply to evaluate) and
`judge` (the question every judge is asked), plus the `expected` label.
Copy `bench/cases/_template.yaml` to add one.

    .venv/bin/python -m bench.run --judges llm decisions -v
    .venv/bin/python -m bench.run --only sw-hint hp-reassure

Judges: `llm` (chat-completion baseline), `decisions` (OpenAI Decisions API, predicate
question thresholded at `DECISIONS_THRESHOLD`, default 0.5), `jev` (TypeSafe's Jev via `typesafe-sdk`, `noul` question thresholded at `JEV_THRESHOLD`, default 0.5; needs `TYPESAFE_API_KEY`).
