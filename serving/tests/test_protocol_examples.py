import json
from pathlib import Path

from serving.service import build_signal, parse_story

EX = json.loads((Path(__file__).resolve().parents[2] / "docs/protocol-examples.json").read_text())


def test_story_example_parses():
    assert parse_story(EX["story"]) == EX["story"]
    for m in EX["poll"]["messages"]:
        parse_story(m)


def test_signal_has_example_keys():
    ex = EX["signals"]["signals"][0]
    assert set(build_signal("s", "AAA", 1, 0.5)) == set(ex)
