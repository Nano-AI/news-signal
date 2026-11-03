import pytest
from hypothesis import given, strategies as st

from serving.model import label_direction


@pytest.mark.parametrize(
    "label,want", [("positive", 1), ("negative", -1), ("neutral", 0), ("Positive", 1), ("NEUTRAL", 0)]
)
def test_mapping(label, want):
    assert label_direction(label) == want


def test_unknown_label_names_it():
    with pytest.raises(ValueError, match="bullish"):
        label_direction("bullish")


@given(st.text())
def test_only_known_labels_map(s):
    if s.lower() in {"positive", "negative", "neutral"}:
        assert label_direction(s) in (-1, 0, 1)
    else:
        with pytest.raises(ValueError):
            label_direction(s)
