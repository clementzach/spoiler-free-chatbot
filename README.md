# Spoiler-free chatbot

Type a spoiler rule, chat normally. Each draft reply is checked by a judge; a draft
that breaks the rule is rejected and regenerated (max 2 attempts), otherwise a safe
fallback is shown. The UI shows a "Spoiler prevented" badge whenever a draft was rejected.

## Run
    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    export OPENAI_API_KEY=...   # optional: MAIN_MODEL, JUDGE_MODEL, DECISIONS_THRESHOLD
    JUDGE=llm|decisions|jev .venv/bin/uvicorn app.main:app --reload

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
