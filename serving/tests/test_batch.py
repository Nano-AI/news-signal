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


def test_bucket_selection():
    from serving.model import pick_bucket

    assert pick_bucket(1, 1) == (1, 16)
    assert pick_bucket(3, 17) == (4, 32)
    assert pick_bucket(33, 64) == (64, 64)
    assert pick_bucket(64, 49) == (64, 64)
    with pytest.raises(ValueError):
        pick_bucket(65, 10)


def test_padding_to_bucket_matches_unpadded(clf):
    from serving.model import pad_to_bucket, pick_bucket

    heads = ["stocks rally", "company earnings beat the guidance rises", "fall"]
    enc = clf.tokenizer(heads, return_tensors="pt", padding=True, truncation=True, max_length=64)
    b, seq = pick_bucket(*enc["input_ids"].shape)
    padded = pad_to_bucket(enc, b, seq, clf.tokenizer.pad_token_id)
    assert padded["input_ids"].shape == (b, seq)
    with torch.inference_mode():
        ref = torch.softmax(clf.model(**enc).logits, -1)
        got = torch.softmax(clf.model(**padded).logits, -1)[: len(heads)]
    assert torch.allclose(ref, got, atol=1e-5)


HEADS50 = [" ".join(["stocks rally fall earnings beat miss the a of as and guidance rises sharply company".split()[(i * 7 + j * 3) % 15]
                     for j in range(1 + (i * 13) % 90)]) for i in range(50)]


def test_arrival_ids_match_padded_tokenizer(clf):
    import numpy as np
    from serving.model import MAX_LENGTH, pick_bucket

    enc = clf.tokenizer(HEADS50, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    assert max(len(clf.tokenize(h)) for h in HEADS50) == MAX_LENGTH
    n, length = enc["input_ids"].shape
    b, seq = pick_bucket(n, length)
    ref = np.zeros((3, b, seq), dtype=np.int64)
    ref[0] = clf.tokenizer.pad_token_id
    ref[0, :n, :length] = enc["input_ids"].numpy()
    ref[2, :n, :length] = enc["attention_mask"].numpy()
    ids_list = [clf.tokenize(h) for h in HEADS50]
    got = np.zeros((3, b, seq), dtype=np.int64)
    got[0] = clf.tokenizer.pad_token_id
    for i, x in enumerate(ids_list):
        got[0, i, : len(x)] = x
        got[2, i, : len(x)] = 1
    assert (got == ref).all()


def test_predict_ids_matches_predict_batch_and_skips_tokenizer(clf, monkeypatch):
    heads = HEADS50[:6]
    ids = [clf.tokenize(h) for h in heads]
    ref = clf.predict_batch(heads)

    def boom(*a, **k):
        raise AssertionError("tokenizer called")

    monkeypatch.setattr(clf, "tokenizer", boom)
    got = clf.predict_ids(ids)
    assert [d for d, _ in got] == [d for d, _ in ref]
    assert all(abs(a - b) < 1e-6 for (_, a), (_, b) in zip(got, ref))
