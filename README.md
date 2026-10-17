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
`DECISIONS_THRESHOLD`, `JEV_THRESHOLD`, `PORT`, and the logging, resilience and judge-window
settings below.

## What the judge sees
Every judge gets the whole conversation (approved turns, the new message and the candidate
reply), not just the last few messages. If the chat outgrows the judge's context window,
the oldest whole messages are dropped and the judge is told how many. The window is
`JUDGE_CONTEXT_TOKENS` (default 100000, a conservative guess; raise it if your judge model
takes more). Tokens are estimated at 3 characters each, so no tokenizer is needed. Trimming is
logged as `judge_history_trimmed`.

## Resilience
- OpenAI and TypeSafe calls time out after `API_TIMEOUT` seconds (default 60) and retry timeouts,
  connection errors, 429s and 5xxs up to `API_MAX_RETRIES` times (default 3) with backoff. Both SDKs
  do this themselves; the app only configures it.
- Wikipedia lookups retry twice on timeouts, 429s and 5xxs. If they still fail, the model is told
  the lookup failed and carries on without it.
- A whole chat turn (generate, judge, regenerate) is capped at `CHAT_TIMEOUT` seconds (default 180).
  The API answers 504 on a timeout and 503 when OpenAI is unreachable, and the UI shows the message.
- If the judge can't be reached after its retries, the reader gets the fallback message, never an
  unchecked draft, and the app doesn't regenerate (a second draft would meet the same outage).

## Logs and the SD card
The app writes JSON lines to stdout only (never to a file), one per event: `startup`, `attempt` (draft and judge
time, verdict, score, reason), `chat_turn` (attempts, blocked, total time), `tool_call`, `retry`,
`judge_error`, `chat_timeout`, `chat_upstream_error`. Messages and replies are not logged unless `LOG_CONTENT=1`, and
then only 200-character snippets. What is always logged: the first 80 characters of the reader's progress
line and the Wikipedia search terms the model used. `/healthz` checks are left out of the access log. `LOG_LEVEL` sets the
verbosity.

Docker keeps container output forever unless told otherwise, so `docker-compose.yml` caps it at 3 files of
5 MB (15 MB total). If you start the container some other way, set the same cap:

    docker run --log-opt max-size=5m --log-opt max-file=3 ...
    # or for every container: /etc/docker/daemon.json
    # {"log-driver": "json-file", "log-opts": {"max-size": "5m", "max-file": "3"}}

If a systemd unit runs `docker compose up` in the foreground, journald also stores the output. Either use
`docker compose up -d` in a `Type=oneshot` / `RemainAfterExit=yes` unit, add `StandardOutput=null` to the unit,
or cap the journal with `SystemMaxUse=50M` in `/etc/systemd/journald.conf`. Read the logs with
`docker compose logs -f app`.

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

## End-to-end evals
`bench/e2e_cases/` runs the whole pipeline the way the server does: the live chat model with live
Wikipedia, the spoiler judge, rejection and regeneration, and approved history carried between turns.
Each case scripts one or more reader messages. Then Jev reads the conversation the reader actually saw
and scores two questions:

- `spoiler_free`: did any assistant reply reveal, confirm, deny or hint at later plot?
- `needs_met`: did the reader get what they needed, as far as possible without spoilers? A flat refusal
  of a safe question fails this, and so does a vague non-answer to a spoilery one.

A case supplies the ground truth for the grader: `spoilers` (what happens after the reader's point) and
`needs` (what a good answer contains). A trial passes when both scores are at least `JEV_THRESHOLD`. The
summary also reports the pipeline's own behaviour: the share of turns where the judge rejected a draft,
the share that ended in the fallback, attempts per turn and seconds per turn. Trials that crash are
counted as errors and left out of the rates. Needs `OPENAI_API_KEY` and `TYPESAFE_API_KEY`. Copy
`bench/e2e_cases/_template.yaml` to add a case.

    .venv/bin/python -m bench.e2e --trials 3 -v
    .venv/bin/python -m bench.e2e --trials 5 --judge decisions --out results.json
    .venv/bin/python -m bench.e2e --only hp-snape-yes-no mockingbird-trial-and-boo

Because Wikipedia is live, results can move with the articles as well as the models. The string-match
tool-use evals above use fixtures when you need a deterministic check.

## Benchmarking judges
Each benchmark is one file in `bench/cases/` with three editable parts:
`context` (reader's progress, history, user message), `response` (the chat reply to evaluate) and
`judge` (the question every judge is asked), plus the `expected` label.
Copy `bench/cases/_template.yaml` to add one.

    .venv/bin/python -m bench.run --judges llm decisions -v
    .venv/bin/python -m bench.run --only sw-hint hp-reassure

Judges: `llm` (chat-completion baseline), `decisions` (OpenAI Decisions API, predicate
question thresholded at `DECISIONS_THRESHOLD`, default 0.5), `jev` (TypeSafe's Jev via `typesafe-sdk`, `noul` question thresholded at `JEV_THRESHOLD`, default 0.5; needs `TYPESAFE_API_KEY`).
