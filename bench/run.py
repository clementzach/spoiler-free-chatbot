"""Benchmark judges on YAML cases in bench/cases/.

Usage: python -m bench.run [--judges llm decisions jev] [--only ID ...] [-v]

Each case file has two editable parts (context, response) plus the `expected`
label. See bench/cases/_template.yaml. Reports per judge: accuracy, missed leaks
(false negatives, the costly error), over-blocks (false positives), errors and
mean latency.
"""
import argparse
import asyncio
import time
from pathlib import Path

import yaml

from app.judges import JUDGES, make_judge

CASE_DIR = Path(__file__).parent / "cases"


def load_cases(only: list[str] | None = None) -> list[dict]:
    cases = []
    for path in sorted(CASE_DIR.glob("*.yaml")):
        if path.name.startswith("_"):
            continue
        c = yaml.safe_load(path.read_text())
        c["id"] = path.stem
        if c["expected"] not in ("violates", "ok"):
            raise ValueError(f"{path.name}: expected must be 'violates' or 'ok'")
        if not only or c["id"] in only:
            cases.append(c)
    return cases


async def run(name: str, cases: list[dict], verbose: bool) -> None:
    judge = make_judge(name)
    tp = tn = fp = fn = errors = 0
    times, misses = [], []
    for c in cases:
        ctx = c["context"]
        t0 = time.perf_counter()
        try:
            v = await judge.judge(
                ctx["progress"], ctx.get("history") or [], ctx["user_message"], c["response"]
            )
        except Exception as exc:
            errors += 1
            print(f"[{name}] {c['id']}: error {type(exc).__name__}: {exc}")
            continue
        times.append(time.perf_counter() - t0)
        want = c["expected"] == "violates"
        if want and v.violates: tp += 1
        elif want: fn += 1; misses.append(c["id"])
        elif v.violates: fp += 1
        else: tn += 1
        if verbose:
            score = f" p={v.score:.2f}" if v.score is not None else ""
            print(f"  {c['id']}: want={c['expected']} got={'violates' if v.violates else 'ok'}{score}")
    n = tp + tn + fp + fn
    print(f"[{name}] n={n} accuracy={(tp + tn) / max(n, 1):.0%} missed_leaks={fn} over_blocks={fp} "
          f"errors={errors} mean_latency={sum(times) / max(len(times), 1):.2f}s")
    if misses:
        print(f"  missed leaks: {', '.join(misses)}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", choices=list(JUDGES), default=list(JUDGES))
    ap.add_argument("--only", nargs="+")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    cases = load_cases(a.only)
    for name in a.judges:
        await run(name, cases, a.verbose)


if __name__ == "__main__":
    asyncio.run(main())
