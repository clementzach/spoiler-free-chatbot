"""Benchmark judges on labelled (rule, response, violates) cases.

Usage: python -m bench.run [llm decisions ...]

Reports, per judge: accuracy, missed leaks (false negatives - the costly error),
over-blocks (false positives), error count and mean latency. A judge that raises
NotImplementedError is skipped.
"""
import asyncio
import json
import sys
import time
from pathlib import Path

from app.judges import make_judge

CASES = [json.loads(l) for l in (Path(__file__).parent / "cases.jsonl").read_text().splitlines() if l.strip()]


async def run(name: str) -> None:
    judge = make_judge(name)
    tp = tn = fp = fn = errors = 0
    times, misses = [], []
    for c in CASES:
        t0 = time.perf_counter()
        try:
            v = await judge.judge(c["rule"], c.get("history", []), c["user"], c["response"])
        except NotImplementedError:
            print(f"[{name}] not implemented, skipping")
            return
        except Exception as exc:
            errors += 1
            print(f"[{name}] {c['id']}: error {type(exc).__name__}: {exc}")
            continue
        times.append(time.perf_counter() - t0)
        if c["violates"] and v.violates: tp += 1
        elif c["violates"]: fn += 1; misses.append(c["id"])
        elif v.violates: fp += 1
        else: tn += 1
    n = tp + tn + fp + fn
    print(f"[{name}] n={n} accuracy={(tp + tn) / max(n, 1):.0%} missed_leaks={fn} over_blocks={fp} "
          f"errors={errors} mean_latency={sum(times) / max(len(times), 1):.2f}s")
    if misses:
        print(f"  missed leaks: {', '.join(misses)}")


async def main() -> None:
    for name in sys.argv[1:] or ["llm", "decisions"]:
        await run(name)


if __name__ == "__main__":
    asyncio.run(main())
