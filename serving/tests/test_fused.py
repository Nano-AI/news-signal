import inspect

import torch
from transformers import BertConfig, BertForSequenceClassification

from serving.model import ARMS, Classifier, FusedBert, median, select_arm


def _model():
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, num_labels=3)
    return BertForSequenceClassification(cfg).eval()


def _batch(lens, S):
    g = torch.Generator().manual_seed(1)
    ids = torch.zeros(len(lens), S, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, n in enumerate(lens):
        ids[i, :n] = torch.randint(1, 100, (n,), generator=g)
        mask[i, :n] = 1
    return ids, torch.zeros_like(ids), mask


def test_fused_matches_hf():
    m = _model()
    fb = FusedBert(m)
    lens = [3, 7, 12, 20, 32, 1, 16, 5, 9, 30] * 2
    ids, tt, mask = _batch(lens, 32)
    with torch.no_grad():
        ref = torch.softmax(m(input_ids=ids, token_type_ids=tt, attention_mask=mask).logits, -1)
        got = fb.forward(ids, tt, mask)
        got2 = fb.forward(ids, tt, mask, sdpa=False)
    assert (ref - got).abs().max() < 1e-4 and (ref - got2).abs().max() < 1e-4
    assert (ref.argmax(-1) == got.argmax(-1)).all()


def test_padding_rows_do_not_change_output():
    fb = FusedBert(_model())
    ids, tt, mask = _batch([6], 16)
    with torch.no_grad():
        alone = fb.forward(ids, tt, mask)
        ids8, tt8, mask8 = _batch([6], 16)
        ids8 = torch.cat([ids8, torch.zeros(7, 16, dtype=torch.long)])
        tt8 = torch.zeros_like(ids8)
        mask8 = torch.cat([mask8, torch.zeros(7, 16, dtype=torch.long)])
        padded = fb.forward(ids8, tt8, mask8)
    assert (alone[0] - padded[0]).abs().max() < 1e-5


def test_cuda_capture_uses_fused_forward():
    src = inspect.getsource(Classifier._capture)
    assert "_capture_arm" in src and "select_arm" in src and "use_sdpa" not in src
    assert "use_sdpa" not in inspect.getsource(FusedBert)


def test_select_arm_min_median_and_ties():
    assert select_arm({"fused_sdpa": 3.0, "fused_explicit": 2.0, "hf_eager": 2.5}) == "fused_explicit"
    assert select_arm({"fused_sdpa": 2.0, "fused_explicit": 2.0, "hf_eager": 2.0}) == "fused_sdpa"
    assert select_arm({"fused_explicit": 2.0, "hf_eager": 2.0}) == "fused_explicit"
    assert select_arm({"hf_eager": 1.0}) == "hf_eager"
    assert set(ARMS) == {"fused_sdpa", "fused_explicit", "hf_eager"}


def test_median():
    assert median([3, 1, 2]) == 2 and median([1, 2, 3, 10]) == 2.5
