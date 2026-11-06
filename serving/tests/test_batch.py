import pytest
import torch
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from serving.model import Classifier


@pytest.fixture(scope="module")
def clf(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    words = "stocks rally fall earnings beat miss the a of as and guidance rises sharply company".split()
    (d / "vocab.txt").write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]))
    BertTokenizerFast(str(d / "vocab.txt")).save_pretrained(d)
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=5 + len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, num_labels=3,
                     id2label={0: "positive", 1: "negative", 2: "neutral"},
                     label2id={"positive": 0, "negative": 1, "neutral": 2})
    BertForSequenceClassification(cfg).save_pretrained(d)
    return Classifier(str(d), "cpu")


def test_batch_matches_single(clf):
    heads = ["stocks rally", "company earnings beat the guidance rises sharply as and of a " * 12,
             "fall", "the company miss", "stocks fall sharply as earnings miss and guidance falls"]
    batch = clf.predict_batch(heads)
    for h, (d, c) in zip(heads, batch):
        d1, c1 = clf.predict(h)
        assert d == d1 and abs(c - c1) < 1e-4
