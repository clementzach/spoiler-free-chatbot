"""Evaluate the chat model's Wikipedia tool use against a stubbed Wikipedia.

Usage: python -m bench.tools [--only ID ...] [--trials N] [-v]

Cases live in bench/tool_cases/ (see _template.yaml). The chat model is live (needs
OPENAI_API_KEY); Wikipedia is a per-case fixture, so tool results are deterministic and
offline. Each case checks the model's calls (looked up when it should, used a sensible
query, stayed on the intro) and substrings of the final reply. Checks are string matches,
so reply_contains_any / reply_excludes are deliberately specific. Models are
nondeterministic: run --trials 3+ and read the pass rate, not one result.
"""
import argparse
import asyncio
import json
from pathlib import Path

import yaml

from app.chat import openai_generate, system_prompt

CASE_DIR = Path(__file__).parent / "tool_cases"
LOOKUP_MODES = ("required", "forbidden", "any")


def load_tool_cases(only: list[str] | None = None) -> list[dict]:
    cases = []
    for path in sorted(CASE_DIR.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        c = yaml.safe_load(path.read_text())
        c["id"] = path.stem
        if c["expect"].get("lookup", "any") not in LOOKUP_MODES:
            raise ValueError(f"{path.name}: expect.lookup must be one of {LOOKUP_MODES}")
        if not only or c["id"] in only:
            cases.append(c)
    return cases


class StubWiki:
    """Drop-in for the live tool runner that serves a case's fixtures and records calls."""

    def __init__(self, spec: dict):
        self.search_text = spec.get("search", "No results.")
        self.error = spec.get("error", False)
        self.pages = {t.lower(): p for t, p in (spec.get("pages") or {}).items()}
        self.calls: list[tuple[str, dict]] = []

    async def __call__(self, name: str, arguments: str) -> str:
        args = json.loads(arguments or "{}")
        self.calls.append((name, args))
        if self.error:
            return "Lookup failed: HTTPStatusError"
        if name == "wikipedia_search":
            return self.search_text
        if name == "wikipedia_page":
            page = self.pages.get(str(args.get("title", "")).lower())
            if page is None:
                return f"No article titled {args.get('title')!r}. Try wikipedia_search."
            return page.get("full", page["intro"]) if args.get("full") else page["intro"]
        return f"Unknown tool {name!r}."


def check(expect: dict, calls: list[tuple[str, dict]], reply: str) -> list[str]:
    """Return the failed expectations (empty = pass)."""
    fails = []
    mode = expect.get("lookup", "any")
    if mode == "required" and not calls:
        fails.append("expected a lookup, made none")
    if mode == "forbidden" and calls:
        fails.append(f"unneeded lookup: {calls[0][0]}({calls[0][1]})")
    queries = [a.get("query") or a.get("title") or "" for _, a in calls]
    terms = expect.get("query_mentions")
    if terms and not any(t.lower() in q.lower() for q in queries for t in terms):
        fails.append(f"no query/title mentions any of {terms}; saw {queries}")
    if expect.get("no_full") and any(n == "wikipedia_page" and a.get("full") for n, a in calls):
        fails.append("fetched a whole article (full=true)")
    if expect.get("page_fetched") and not any(n == "wikipedia_page" for n, _ in calls):
        fails.append("never fetched an article")
    low = reply.lower()
    wanted = expect.get("reply_contains_any")
    if wanted and not any(s.lower() in low for s in wanted):
        fails.append(f"reply has none of {wanted}")
    for s in expect.get("reply_excludes") or []:
        if s.lower() in low:
            fails.append(f"reply contains {s!r}")
    return fails


async def run_trial(case: dict, client=None, model: str | None = None) -> tuple[list[str], list, str]:
    stub = StubWiki(case["wiki"])
    ctx = case["context"]
    messages = [
        {"role": "system", "content": system_prompt(ctx["progress"])},
        *(ctx.get("history") or []),
        {"role": "user", "content": ctx["user_message"]},
    ]
    reply = await openai_generate(client, model, run_tool=stub)(messages)
    return check(case["expect"], stub.calls, reply), stub.calls, reply


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    passed = total = 0
    for c in load_tool_cases(a.only):
        ok, failures = 0, []
        for _ in range(a.trials):
            try:
                fails, calls, reply = await run_trial(c)
            except Exception as exc:
                fails, calls, reply = [f"error {type(exc).__name__}: {exc}"], [], ""
            ok += not fails
            failures += fails
            if a.verbose:
                print(f"  calls={calls}\n  reply={reply!r}")
        passed, total = passed + ok, total + a.trials
        print(f"{c['id']}: {ok}/{a.trials}")
        for f in dict.fromkeys(failures):
            print(f"    - {f}")
    print(f"tool use: {passed}/{total} trials passed ({passed / max(total, 1):.0%})")


if __name__ == "__main__":
    asyncio.run(main())
