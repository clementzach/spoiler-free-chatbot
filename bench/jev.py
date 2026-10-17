"""Shared plumbing for the benchmarks that have TypeSafe's Jev grade a transcript."""
import os

from typesafe_sdk import AsyncTypeSafeClient, Noul

from app.resilience import typesafe_client

# question name -> (instructions, what a true answer means, what a false answer means)
Questions = dict[str, tuple[str, str, str]]


class JevGrader:
    """One `system_one` call per transcript; each question is a noul scoring P(the behaviour
    was good). A transcript passes a question when its score is at least `threshold`."""

    def __init__(self, client: AsyncTypeSafeClient | None = None, threshold: float | None = None):
        self._client = client
        self.threshold = threshold if threshold is not None else float(
            os.getenv("JEV_THRESHOLD", "0.5")
        )

    @property
    def client(self) -> AsyncTypeSafeClient:
        if self._client is None:  # lazy: constructing it requires the API key
            self._client = typesafe_client()
        return self._client

    async def score(self, state: str, questions: Questions) -> dict[str, float]:
        resp = await self.client.system_one(
            state=state,
            questions={
                name: Noul(instructions=q, criteria={"true": good, "false": bad})
                for name, (q, good, bad) in questions.items()
            },
        )
        return {name: float(resp.nouls[name].noul) for name in questions}

    def failed(self, scores: dict[str, float]) -> list[str]:
        return [n for n, p in scores.items() if p < self.threshold]
