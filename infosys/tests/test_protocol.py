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
