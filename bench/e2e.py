"""End-to-end evaluation of the whole chat pipeline, graded by Jev.

Usage: python -m bench.e2e [--only ID ...] [--trials N] [--concurrency N] [--judge NAME]
                           [--out FILE] [-v]

Each case in bench/e2e_cases/ (see _template.yaml) is a scripted conversation. It runs
through the real `respond()` loop exactly as the server does: the live chat model with live
Wikipedia lookups, the spoiler judge chosen by --judge / $JUDGE, rejection and regeneration,
and approved history carried from turn to turn. Needs OPENAI_API_KEY, and TYPESAFE_API_KEY
for the grader (and for the judge when --judge jev).

Jev then reads the transcript the reader actually saw and answers two questions:
  spoiler_free  did any assistant reply reveal, confirm, deny or hint at later plot?
  needs_met     did the reader get what they needed, as far as possible without spoilers?
A case supplies the ground truth: `spoilers` (what happens after the reader's point) and
`needs` (what a good answer contains). A trial passes when both scores are >= JEV_THRESHOLD
(default 0.5). Models are nondeterministic: use --trials 3+ and read the rates.

The summary also reports how the pipeline behaved: how many turns the judge rejected a
draft, how many ended in the fallback (blocked), attempts per turn and seconds per turn.
Trials that crash (API outage, grader error) are counted as errors and left out of the rates.
"""
import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from app.chat import openai_generate, respond
from app.judges import make_judge

from .jev import JevGrader, Questions

CASE_DIR = Path(__file__).parent / "e2e_cases"


def load_e2e_cases(only: list[str] | None = None) -> list[dict]:
    cases = []
    for path in sorted(CASE_DIR.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        c = yaml.safe_load(path.read_text())
        c["id"] = path.stem
        for key in ("turns", "needs", "spoilers"):
            if not c.get(key):
                raise ValueError(f"{path.name}: missing {key!r}")
        if not only or c["id"] in only:
            cases.append(c)
    return cases


@dataclass
class TurnResult:
    user: str
    reply: str
    attempts: int
    spoiler_prevented: bool  # the judge rejected at least one draft
    blocked: bool  # every attempt was rejected; the reply is the fallback
    seconds: float


@dataclass
class Result:
    case: str
    turns: list[TurnResult] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)  # question -> P(good)
    error: str | None = None


async def run_trial(case: dict, generate, judge) -> list[TurnResult]:
    """Play the scripted user messages through `respond`, carrying approved history like the UI."""
    progress = case["context"]["progress"]
    history: list[dict] = []
    turns = []
    for message in case["turns"]:
        t0 = time.perf_counter()
        r = await respond(generate, judge, progress, history, message)
        turns.append(TurnResult(message, r.reply, r.attempts, r.spoiler_prevented, r.blocked,
                                time.perf_counter() - t0))
        history += [{"role": "user", "content": message}, {"role": "assistant", "content": r.reply}]
    return turns


# --- Jev grading: P(true) = the behaviour was good ---

E2E_QUESTIONS: Questions = {
    "spoiler_free": (
        "Does any ASSISTANT message in the conversation reveal, confirm, deny, or hint at a "
        "plot detail beyond the reader's progress? The LATER PLOT list is ground truth for what "
        "the reader has not reached yet; also apply your own knowledge of the work. Yes/no "
        "answers about later events, reassurance ('don't worry, she's fine'), foreshadowing "
        "('pay attention to...') and saying a mystery 'will be explained later' all count as "
        "spoilers. Facts the reader has already reached, outside historical or author context, "
        "and how the world works as shown so far do not.",
        "No assistant message contains a spoiler",
        "An assistant message reveals or hints at later plot",
    ),
    "needs_met": (
        "Did the assistant give the reader what they needed, as far as possible without "
        "spoilers? WHAT THE READER NEEDS says what a good reply contains. When the honest "
        "answer would be a spoiler, a good reply says so in a sentence and still helps with "
        "the safe part. A generic refusal, an evasive or off-topic reply, withholding "
        "things that are safe to say, or stating wrong facts is not good. Judge every user "
        "turn in the conversation.",
        "The reader got a helpful, accurate, responsive answer without a spoiler",
        "The reply was unhelpful, evasive, over-cautious, off-topic or wrong",
    ),
}


def render_conversation(case: dict, turns: list[TurnResult]) -> str:
    """What the grader sees: the ground truth, then the conversation exactly as the reader saw it."""
    lines = [
        f"READER PROGRESS: {case['context']['progress']}",
        "",
        f"WHAT THE READER NEEDS: {' '.join(case['needs'].split())}",
        "",
        "LATER PLOT (the reader has NOT reached any of this):",
        *[f"- {s}" for s in case["spoilers"]],
        "",
        "CONVERSATION:",
    ]
    for t in turns:
        lines += [f"USER: {t.user}", f"ASSISTANT: {t.reply}"]
    if case.get("notes"):
        lines += ["", f"EVALUATOR NOTES: {' '.join(case['notes'].split())}"]
    return "\n".join(lines)


class JevE2EGrader(JevGrader):
    async def grade(self, case: dict, turns: list[TurnResult]) -> dict[str, float]:
        return await self.score(render_conversation(case, turns), E2E_QUESTIONS)


async def evaluate(case: dict, generate, judge, grader: JevE2EGrader, sem: asyncio.Semaphore) -> Result:
    result = Result(case["id"])
    async with sem:
        try:
            result.turns = await run_trial(case, generate, judge)
        except Exception as exc:
            result.error = f"pipeline error {type(exc).__name__}: {exc}"
            return result
        try:
            result.scores = await grader.grade(case, result.turns)
        except Exception as exc:
            result.error = f"grader error {type(exc).__name__}: {exc}"
    return result


def summarize(results: list[Result], threshold: float) -> dict:
    graded = [r for r in results if r.scores]
    turns = [t for r in results for t in r.turns]

    def rate(count: int, total: int) -> float | None:
        return count / total if total else None

    def passing(r: Result, q: str) -> bool:
        return r.scores[q] >= threshold

    return {
        "trials": len(results),
        "errors": sum(1 for r in results if r.error),
        "spoiler_free": rate(sum(passing(r, "spoiler_free") for r in graded), len(graded)),
        "needs_met": rate(sum(passing(r, "needs_met") for r in graded), len(graded)),
        "both": rate(sum(passing(r, "spoiler_free") and passing(r, "needs_met") for r in graded), len(graded)),
        "turns": len(turns),
        "turns_with_rejected_draft": rate(sum(t.spoiler_prevented for t in turns), len(turns)),
        "turns_blocked": rate(sum(t.blocked for t in turns), len(turns)),
        "mean_attempts": sum(t.attempts for t in turns) / len(turns) if turns else None,
        "mean_seconds_per_turn": sum(t.seconds for t in turns) / len(turns) if turns else None,
    }


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def print_summary(s: dict) -> None:
    print(f"\nend to end: {s['trials']} trials, {s['errors']} errors")
    print(f"  spoiler-free {pct(s['spoiler_free'])}   needs met {pct(s['needs_met'])}   "
          f"both {pct(s['both'])}")
    if s["turns"]:
        print(f"  {s['turns']} turns: {pct(s['turns_with_rejected_draft'])} had a draft rejected by the "
              f"judge, {pct(s['turns_blocked'])} ended in the fallback, "
              f"{s['mean_attempts']:.2f} attempts and {s['mean_seconds_per_turn']:.1f}s per turn")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--judge", help="spoiler judge in the pipeline (default $JUDGE or llm)")
    ap.add_argument("--out", help="write per-trial transcripts, scores and the summary as JSON")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    cases = load_e2e_cases(a.only)
    generate, judge, grader = openai_generate(), make_judge(a.judge), JevE2EGrader()
    sem = asyncio.Semaphore(a.concurrency)
    results = await asyncio.gather(*(
        evaluate(c, generate, judge, grader, sem) for c in cases for _ in range(a.trials)
    ))

    for case in cases:
        mine = [r for r in results if r.case == case["id"]]
        graded = [r for r in mine if r.scores]
        counts = {q: sum(r.scores[q] >= grader.threshold for r in graded) for q in E2E_QUESTIONS}
        print(f"{case['id']}: " + "  ".join(f"{q} {n}/{len(mine)}" for q, n in counts.items()))
        for r in mine:
            bad = grader.failed(r.scores)
            if r.error:
                print(f"    - {r.error}")
            if bad or a.verbose:
                for q in bad:
                    print(f"    - {q} (p={r.scores[q]:.2f})")
                for t in r.turns:
                    flags = f" [rejected drafts, {t.attempts} attempts]" if t.spoiler_prevented else ""
                    print(f"      USER: {t.user}\n      BOT{flags}: {t.reply}")
    summary = summarize(results, grader.threshold)
    print_summary(summary)
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"summary": summary, "threshold": grader.threshold, "trials": [asdict(r) for r in results]},
            indent=2))


if __name__ == "__main__":
    asyncio.run(main())
