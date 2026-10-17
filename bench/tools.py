"""Evaluate the chat model's Wikipedia tool use against a stubbed Wikipedia.

Usage: python -m bench.tools [--only ID ...] [--trials N] [--jev] [-v]

Cases live in bench/tool_cases/ (see _template.yaml). The chat model is live (needs
OPENAI_API_KEY); Wikipedia is a per-case fixture, so tool results are deterministic and
offline. Each case checks the model's calls (looked up when it should, used a sensible
query, stayed on the intro) and substrings of the final reply. Checks are string matches,
so reply_contains_any / reply_excludes are deliberately specific. Models are
nondeterministic: run --trials 3+ and read the pass rate, not one result.

--jev also has TypeSafe's Jev grade each transcript (tool calls, tool results, final reply)
on four questions the string checks can't answer well: were lookups warranted, were the
queries relevant, is the reply grounded in what the tools returned, is it spoiler-free.
Needs TYPESAFE_API_KEY. A case can add a `notes:` string to steer the grader.
"""
import argparse
import asyncio
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from typesafe_sdk import AsyncTypeSafeClient, Noul

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
        self.results: list[str] = []  # parallel to calls

    async def __call__(self, name: str, arguments: str) -> str:
        args = json.loads(arguments or "{}")
        result = self._serve(name, args)
        self.calls.append((name, args))
        self.results.append(result)
        return result

    def _serve(self, name: str, args: dict) -> str:
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


@dataclass
class Trial:
    fails: list[str]
    calls: list[tuple[str, dict]]
    results: list[str]
    reply: str
    jev: dict[str, float] = field(default_factory=dict)  # question -> P(true); filled by --jev


async def run_trial(case: dict, client=None, model: str | None = None) -> Trial:
    stub = StubWiki(case["wiki"])
    ctx = case["context"]
    messages = [
        {"role": "system", "content": system_prompt(ctx["progress"])},
        *(ctx.get("history") or []),
        {"role": "user", "content": ctx["user_message"]},
    ]
    reply = await openai_generate(client, model, run_tool=stub)(messages)
    return Trial(check(case["expect"], stub.calls, reply), stub.calls, stub.results, reply)


# --- Jev grading: one call per trial, four noul questions, each P(the behaviour was good) ---

MAX_RESULT_CHARS = 3000

JEV_QUESTIONS = {
    "lookup_appropriate": (
        "Did the assistant use Wikipedia lookups appropriately? It should look up factual "
        "questions about the work (cast, dates, authorship, background) rather than rely on "
        "memory, and should not look anything up for casual chat or opinions.",
        "Lookups were made when warranted and skipped when not",
        "The assistant skipped a needed lookup, or looked something up needlessly",
    ),
    "queries_relevant": (
        "Were the search queries and article titles relevant to the work the reader is on "
        "(resolving it from the reader's progress if the question didn't name it)? Fetching "
        "the whole article (full=true) is only justified when the intro and snippets could "
        "not answer the question, since full articles contain plot sections. If no lookups "
        "were made, answer true.",
        "Queries targeted the right work and the assistant fetched only as much as needed",
        "Queries were off-target, or a whole article was fetched without need",
    ),
    "grounded": (
        "Is every factual claim in the final reply supported by the tool results, or else "
        "clearly hedged? If lookups failed or returned nothing, the reply must not present "
        "invented facts as certain.",
        "The reply's facts are supported by the tool results or honestly hedged",
        "The reply states facts the tools did not support, or contradicts them",
    ),
    "spoiler_free": (
        "Does the final reply avoid revealing, confirming or hinting at any plot detail past "
        "the reader's progress, including details that appear in the tool results? Under "
        "the spoiler policy, outside facts such as cast, dates and authorship are fine.",
        "The reply contains no spoilers for this reader",
        "The reply reveals or hints at a plot detail past the reader's progress",
    ),
}


def render_transcript(case: dict, trial: Trial) -> str:
    ctx = case["context"]
    lines = [f"READER PROGRESS: {ctx['progress']}", "", "CONVERSATION:"]
    lines += [f"{m['role'].upper()}: {m['content']}" for m in ctx.get("history") or []]
    lines += [f"USER: {ctx['user_message']}", "", "TOOLS AVAILABLE: wikipedia_search(query), "
              "wikipedia_page(title, full=false)", "", "TOOL CALLS:"]
    if not trial.calls:
        lines.append("(none)")
    for i, ((name, args), res) in enumerate(zip(trial.calls, trial.results), 1):
        lines.append(f"{i}. {name}({json.dumps(args)}) -> {res[:MAX_RESULT_CHARS]}")
    lines += ["", f"FINAL REPLY:\n{trial.reply}"]
    if case.get("notes"):
        lines += ["", f"EVALUATOR NOTES: {case['notes']}"]
    return "\n".join(lines)


class JevToolGrader:
    def __init__(self, client: AsyncTypeSafeClient | None = None, threshold: float | None = None):
        self._client = client
        self.threshold = threshold if threshold is not None else float(
            os.getenv("JEV_THRESHOLD", "0.5")
        )

    @property
    def client(self) -> AsyncTypeSafeClient:
        if self._client is None:  # lazy: constructing it requires the API key
            self._client = AsyncTypeSafeClient(model="jev-latest")
        return self._client

    async def grade(self, case: dict, trial: Trial) -> dict[str, float]:
        resp = await self.client.system_one(
            state=render_transcript(case, trial),
            questions={
                name: Noul(instructions=q, criteria={"true": good, "false": bad})
                for name, (q, good, bad) in JEV_QUESTIONS.items()
            },
        )
        return {name: float(resp.nouls[name].noul) for name in JEV_QUESTIONS}

    def failed(self, scores: dict[str, float]) -> list[str]:
        return [n for n, p in scores.items() if p < self.threshold]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--jev", action="store_true", help="also grade transcripts with Jev")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    grader = JevToolGrader() if a.jev else None
    passed = jev_passed = total = 0
    for c in load_tool_cases(a.only):
        ok = jev_ok = 0
        failures: list[str] = []
        jev_failures: list[str] = []
        for _ in range(a.trials):
            try:
                t = await run_trial(c)
            except Exception as exc:
                t = Trial([f"error {type(exc).__name__}: {exc}"], [], [], "")
            ok += not t.fails
            failures += t.fails
            if grader and t.reply:
                try:
                    t.jev = await grader.grade(c, t)
                    bad = grader.failed(t.jev)
                    jev_ok += not bad
                    jev_failures += [f"{n} (p={t.jev[n]:.2f})" for n in bad]
                except Exception as exc:
                    jev_failures.append(f"grader error {type(exc).__name__}: {exc}")
            if a.verbose:
                print(f"  calls={t.calls}\n  reply={t.reply!r}" + (f"\n  jev={t.jev}" if t.jev else ""))
        passed, jev_passed, total = passed + ok, jev_passed + jev_ok, total + a.trials
        print(f"{c['id']}: checks {ok}/{a.trials}" + (f"  jev {jev_ok}/{a.trials}" if grader else ""))
        for f in dict.fromkeys(failures):
            print(f"    - {f}")
        for f in dict.fromkeys(jev_failures):
            print(f"    - jev: {f}")
    print(f"tool use: {passed}/{total} trials passed checks ({passed / max(total, 1):.0%})")
    if grader:
        print(f"          {jev_passed}/{total} passed all Jev questions ({jev_passed / max(total, 1):.0%})")


if __name__ == "__main__":
    asyncio.run(main())
