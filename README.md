# Spoiler-free chatbot

Type a spoiler rule, chat normally. Each draft reply is checked by a judge; a draft
that breaks the rule is rejected and regenerated (max 2 attempts), otherwise a safe
fallback is shown. The UI shows a "Spoiler prevented" badge whenever a draft was rejected.

## Run
    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    export OPENAI_API_KEY=...   # optional: MAIN_MODEL, JUDGE_MODEL, JUDGE=llm|decisions
    .venv/bin/uvicorn app.main:app --reload

## Test / benchmark
    .venv/bin/pytest
    .venv/bin/python -m bench.run llm decisions   # compares judges on bench/cases.jsonl

No server state: the browser sends the approved history and the current rule each turn.
