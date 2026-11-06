import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from serving.model import ARMS, FusedBert, load_kernels
from serving.tests.test_fused import _batch, _model

SRC = Path(__file__).parents[1] / "kernels" / "bert_fused.hip"


def np_add_ln(x, h, g, b, eps):
    v = (x + h).astype(np.float64)
    mu = v.mean(-1, keepdims=True)
    var = ((v - mu) ** 2).mean(-1, keepdims=True)
    return (v - mu) / np.sqrt(var + eps) * g + b


def np_attn(q, k, v, mask):
    """q,k,v: [S,d] one head; mask [S] (1 = keep). Additive -1e4 key mask, softmax over keys."""
    s = q @ k.T / np.sqrt(q.shape[-1]) + np.where(mask, 0.0, -1e4)[None]
    p = np.exp(s - s.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)) @ v


def np_emb_ln(word, pos, tok, ids, tt, g, b, eps):
    S = ids.shape[1]
    return np_add_ln(word[ids] + pos[:S], tok[tt], g, b, eps)  # LN(word+pos+tok), no residual


def test_numpy_reference_matches_torch_math():
    r = np.random.default_rng(0)
    x, h = r.normal(size=(5, 32)), r.normal(size=(5, 32))
    g, b = r.normal(size=32), r.normal(size=32)
    ref = F.layer_norm(torch.tensor(x + h), (32,), torch.tensor(g), torch.tensor(b), 1e-12).numpy()
    assert np.abs(np_add_ln(x, h, g, b, 1e-12) - ref).max() < 1e-9
    q, k, v = (r.normal(size=(7, 16)) for _ in range(3))
    mask = np.array([1, 1, 1, 0, 1, 0, 0], bool)
    m = torch.tensor(np.where(mask, 0.0, -1e4))
    ref = F.scaled_dot_product_attention(*(torch.tensor(t) for t in (q, k, v)), attn_mask=m).numpy()
    assert np.abs(np_attn(q, k, v, mask) - ref).max() < 1e-9


def test_emb_ln_reference_matches_fused_embedding():
    m = _model()
    fb = FusedBert(m)
    ids, tt, _ = _batch([3, 5], 8)
    ref = F.layer_norm(fb.word[ids] + fb.pos[:8] + fb.tok[tt], (32,), *fb.emb_ln, fb.eps).detach().numpy()
    got = np_emb_ln(fb.word.numpy(), fb.pos.numpy(), fb.tok.numpy(), ids.numpy(), tt.numpy(),
                    fb.emb_ln[0].detach().numpy(), fb.emb_ln[1].detach().numpy(), fb.eps)
    assert np.abs(ref - got).max() < 1e-4


def test_hip_arm_registered_first_and_absent_lib_is_none(tmp_path):
    assert ARMS[0] == "fused_hip"
    assert load_kernels(tmp_path) is None and load_kernels(None) is None


def test_kernel_source_has_launchers():
    s = SRC.read_text()
    for name in ("launch_emb_ln", "launch_add_ln", "launch_attn"):
        assert f'int {name}(' in s


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("b,seq", [(1, 16), (4, 48), (64, 64)])
def test_hip_arm_matches_fused_explicit(tmp_path, b, seq):
    so = tmp_path / "bert_fused.so"
    subprocess.run(["hipcc", "-O3", "--offload-arch=gfx90a", "-shared", "-fPIC", str(SRC), "-o", str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    torch.manual_seed(0)
    from transformers import BertConfig, BertForSequenceClassification
    cfg = BertConfig(vocab_size=500, hidden_size=768, num_hidden_layers=2, num_attention_heads=12,
                     intermediate_size=3072, num_labels=3)
    fb = FusedBert(BertForSequenceClassification(cfg).eval().half().cuda())
    g = torch.Generator().manual_seed(1)
    lens = torch.randint(1, seq + 1, (b,), generator=g)
    mask = (torch.arange(seq)[None] < lens[:, None]).long()
    ids = (torch.randint(1, 500, (b, seq), generator=g) * mask).cuda()
    tt, mask = torch.zeros_like(ids), mask.cuda()
    with torch.no_grad():
        ref = fb.forward(ids, tt, mask, sdpa=False)
        got = fb.forward_hip(lib, ids, tt, mask)
    assert not torch.isnan(got).any()
    assert (ref - got).abs().max() < 1e-2
    assert (ref.argmax(-1) == got.argmax(-1)).all()
