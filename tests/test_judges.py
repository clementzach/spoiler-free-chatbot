from types import SimpleNamespace as NS

import pytest

from app.judges import (CHARS_PER_TOKEN, RESERVED_TOKENS, DecisionsJudge, LLMJudge, fit_history,
                        history_room, render_context, state)


def chat(n, size=10):
    out = []
    for i in range(n):
        out += [{"role": "user", "content": f"q{i} " + "x" * size},
                {"role": "assistant", "content": f"a{i} " + "y" * size}]
    return out


def test_whole_history_is_kept_when_it_fits():
    h = chat(30)
    assert fit_history(h, None) == h
    assert fit_history(h, 10_000) == h
    text = render_context(h, "now?", "draft", 10_000)
    assert "q0 " in text and "a29 " in text and "omitted" not in text
    assert text.index("q0 ") < text.index("a29 ") < text.index("USER: now?") < text.index("draft")


def test_oldest_whole_messages_are_dropped_first_and_the_gap_is_stated(events):
    h = chat(10)
    kept = fit_history(h, 100)
    assert kept == h[-len(kept):] and 0 < len(kept) < len(h)
    assert all(len(f"{m['role'].upper()}: {m['content']}") + 1 <= 100 for m in kept[:1])
    text = render_context(h, "now?", "draft", 100)
    assert f"[{len(h) - len(kept)} earlier messages omitted" in text and "q0 " not in text
    assert events[0][0] == "judge_history_trimmed" and events[0][2]["dropped"] == len(h) - len(kept)


def test_no_room_means_no_history_but_the_question_and_candidate_survive():
    text = render_context(chat(3), "now?", "draft", 0)
    assert "q0" not in text and "USER: now?" in text and "CANDIDATE REPLY:\ndraft" in text


def test_history_room_scales_with_the_window_and_subtracts_the_fixed_parts():
    assert history_room(RESERVED_TOKENS + 100, "a" * 50) == 100 * CHARS_PER_TOKEN - 50
    assert history_room(10, "a" * 10_000) == 0


def test_api_judges_get_the_full_chat_in_their_input():
    h = chat(40, size=200)  # ~17k chars: far more than the old 4-message window
    text = state("ch. 3", h, "now?", "draft")
    assert "q0 " in text and "a39 " in text and "omitted" not in text


def test_a_small_window_trims_the_state_not_the_policy():
    h = chat(40, size=200)
    text = state("ch. 3", h, "now?", "draft", window_tokens=RESERVED_TOKENS + 4_000)
    assert "omitted" in text and "q0 " not in text and "a39 " in text
    assert "Never reveal plot details" in text and "Reader's progress: ch. 3" in text


@pytest.mark.asyncio
async def test_llm_judge_sends_every_turn():
    sent = []

    async def create(**kw):
        sent.append(kw["messages"])
        return NS(choices=[NS(message=NS(content='{"violates": false, "reason": ""}'))])

    j = LLMJudge(client=NS(chat=NS(completions=NS(create=create))), window_tokens=100_000)
    h = chat(25, size=100)
    await j.judge("ch. 3", h, "now?", "draft")
    prompt = sent[0][1]["content"]
    assert all(m["content"] in prompt for m in h)


@pytest.mark.asyncio
async def test_llm_judge_trims_to_its_window():
    sent = []

    async def create(**kw):
        sent.append(kw["messages"])
        return NS(choices=[NS(message=NS(content='{"violates": false, "reason": ""}'))])

    j = LLMJudge(client=NS(chat=NS(completions=NS(create=create))), window_tokens=RESERVED_TOKENS + 3_000)
    h = chat(100, size=100)
    await j.judge("ch. 3", h, "now?", "draft")
    prompt = sent[0][1]["content"]
    assert "omitted" in prompt and h[0]["content"] not in prompt and h[-1]["content"] in prompt


@pytest.mark.asyncio
async def test_decisions_judge_input_contains_the_whole_chat():
    seen = {}

    async def create(**kw):
        seen.update(kw)
        return NS(answers=[NS(type="predicate", probability=0.1)])

    j = DecisionsJudge(client=NS(decisions=NS(create=create)), threshold=0.5, window_tokens=100_000)
    h = chat(25, size=100)
    await j.judge("ch. 3", h, "now?", "draft")
    assert all(m["content"] in seen["input"] for m in h)
