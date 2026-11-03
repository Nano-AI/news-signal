import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from infosys.protocol import (
    EndMessage,
    FeedList,
    PollResponse,
    SignalBatch,
    SignalReply,
    Status,
    StoryMessage,
    TrainRecord,
)

EXAMPLES = json.loads((Path(__file__).resolve().parents[2] / "docs" / "protocol-examples.json").read_text())
MODELS = {
    "feeds": FeedList,
    "story": StoryMessage,
    "end": EndMessage,
    "poll": PollResponse,
    "signals": SignalBatch,
    "signals_reply": SignalReply,
    "status": Status,
    "train_record": TrainRecord,
}


@pytest.mark.parametrize("key", MODELS)
def test_examples_validate(key):
    MODELS[key].model_validate(EXAMPLES[key])


def test_examples_have_exactly_the_documented_keys():
    assert set(EXAMPLES) == set(MODELS) | {"invalid_signals"}


@pytest.mark.parametrize("case", EXAMPLES["invalid_signals"], ids=lambda c: str(c["key"]))
def test_invalid_signals_name_the_key(case):
    with pytest.raises(ValidationError) as err:
        SignalBatch.model_validate(case["body"])
    assert any(case["key"] in e["loc"] for e in err.value.errors())


def _story(**over):
    return {**EXAMPLES["story"], **over}


def test_impossible_timestamp_rejected():
    for bad in ("2023-13-45T99:99:99Z", "2023-07-10 14:31:05", "2023-02-30T10:00:00Z"):
        with pytest.raises(ValidationError):
            StoryMessage.model_validate(_story(published_at=bad))


def test_unknown_story_is_not_a_rejection_reason():
    from infosys.protocol import Rejection

    with pytest.raises(ValidationError):
        Rejection(index=0, reason="unknown_story")


def test_train_record_returns_keys_equal_tickers():
    rec = EXAMPLES["train_record"]
    for bad in ({**rec, "returns": {"AAA": 0.1}}, {**rec, "returns": {**rec["returns"], "ZZZ": 0.1}}):
        with pytest.raises(ValidationError):
            TrainRecord.model_validate(bad)


def test_examples_are_mutually_consistent():
    story, batch, reply = EXAMPLES["story"], EXAMPLES["signals"], EXAMPLES["signals_reply"]
    by_index = {r["index"]: r["reason"] for r in reply["rejected"]}
    for i, sig in enumerate(batch["signals"]):
        known = {story["story_id"]: story["tickers"]}.get(sig["story_id"])
        if sig["story_id"] == story["story_id"]:
            assert (sig["ticker"] not in known) == (by_index.get(i) == "unknown_ticker")
    assert reply["accepted"] + len(reply["rejected"]) == len(batch["signals"])
