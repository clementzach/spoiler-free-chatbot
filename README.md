# Spoiler-free chatbot

Say what you're reading and how far you've got, then chat normally. A fixed spoiler
policy (`app/policy.py`) governs both the chat model and the judge: nothing past your
point in the plot, but obvious facts, outside historical context and answers from your
current understanding are fine, and when unsure the bot asks what you already know.
Each draft reply is checked by a judge, which sees the whole conversation. A draft that
spoils is never shown or retried; the reader is told to ask a different question. If the
model or the judge fails, the UI shows "Something went wrong" and nothing unchecked is
shown. If the conversation outgrows the judge's input limit, the UI asks the reader to
start a new chat rather than judging a truncated conversation.

## Run locally
    python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
    cp .env.example .env   # fill in keys, then: set -a; . ./.env; set +a
    .venv/bin/uvicorn app.main:app --reload

## Configuration
See `.env.example`. `OPENAI_API_KEY` is always required (chat model); `TYPESAFE_API_KEY`
is also required when `JUDGE=jev`. The app refuses to start if a required key is missing.
Optional: `JUDGE` (`llm` default, `decisions`, `jev`; see `JUDGES` in `app/judges.py`),
`MAIN_MODEL`, `JUDGE_MODEL`, `DECISIONS_THRESHOLD`, `JEV_THRESHOLD`, `PORT`, and
`REQUEST_TIMEOUT` (seconds per model call attempt, default 30) and `MAX_RETRIES` (default 1)
for every chat-model and judge call, and `JUDGE_MAX_INPUT_TOKENS` to override the selected judge's input limit (defaults: 1,047,576
for `llm`, 1,050,000 for `decisions`, 32,000 for `jev`). Input size is estimated at 3
characters per token, which overcounts on purpose.

## Dependencies
`requirements.in` / `requirements-dev.in` list direct dependencies; the `.txt` files are
pinned lockfiles generated with [pip-tools](https://github.com/jazzband/pip-tools) on
Python 3.13 (the Docker image's version). After editing an `.in` file:

    pip-compile --strip-extras --no-emit-index-url -o requirements.txt requirements.in
    pip-compile --strip-extras --no-emit-index-url -o requirements-dev.txt requirements-dev.in

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

No server state: the browser sends the approved history and the reader's progress each
turn, and keeps the chat in `localStorage` so a reload doesn't lose it ("New chat" clears
it). Changing your progress stops earlier messages being sent to the bot.

## Benchmarking judges
Each benchmark is one file in `bench/cases/` with two editable parts:
`context` (reader's progress, history, user message) and `response` (the chat reply to
evaluate), plus the `expected` label.
Copy `bench/cases/_template.yaml` to add one.

    .venv/bin/python -m bench.run --judges llm decisions -v
    .venv/bin/python -m bench.run --only sw-hint hp-reassure

Judges: `llm` (chat-completion baseline), `decisions` (OpenAI Decisions API, predicate
question thresholded at `DECISIONS_THRESHOLD`, default 0.5), `jev` (TypeSafe's Jev via `typesafe-sdk`, `noul` question thresholded at `JEV_THRESHOLD`, default 0.5; needs `TYPESAFE_API_KEY`).
