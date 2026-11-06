"""Native runtime: binary tokenizer parity and an end-to-end run against a fake newsd (CPU debug mode)."""

import asyncio
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from aiohttp import web
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from serving import native
from serving.model import MAX_LENGTH
from test_service import FakeNewsd

HEADLINES = ["up up", "down down", "flat flat", "Café UP 日本!", "up down\tflat", "UPS unknown wörd ##s"]


@pytest.fixture(scope="module")
def prepared():
    tmp = tempfile.mkdtemp()
    try:
        if os.environ.get("NATIVE_TEST_BIN"):  # editor without a ROCm-torch Python: use a binary built elsewhere
            (Path(tmp) / "native").mkdir()
            exe = Path(tmp) / "native" / native.BINARY
            shutil.copy(os.environ["NATIVE_TEST_BIN"], exe)
        else:
            exe = native.build(tmp)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"native build unavailable: {e}")
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "up", "down", "flat", "##s", "!", "cafe", "unknown", "word",
             "u", "##p", "o", "w", "##r", "##d", "ups"] + list("abcdefghijklmnopqrstuvwxyz")
    words = list(dict.fromkeys(words))
    mdir = Path(tmp) / "model"
    mdir.mkdir()
    (mdir / "vocab.txt").write_text("\n".join(words) + "\n")
    BertTokenizerFast(str(mdir / "vocab.txt"), do_lower_case=True).save_pretrained(mdir)
    torch.manual_seed(0)
    cfg = BertConfig(vocab_size=len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=128, num_labels=3,
                     id2label={0: "positive", 1: "negative", 2: "neutral"},
                     label2id={"positive": 0, "negative": 1, "neutral": 2})
    BertForSequenceClassification(cfg).save_pretrained(mdir)
    native.export_model(str(mdir), tmp)
    return tmp, str(mdir), exe


def test_tokenizer_parity(prepared):
    out, model, _ = prepared
    ok, n = native.tokenizer_parity(model, out, HEADLINES + native._SYNTH)
    assert ok == n


async def _scenario(out, model, exe):
    fake = FakeNewsd()
    runner = web.AppRunner(fake.app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        p = await asyncio.create_subprocess_exec(
            str(exe), "--newsd", f"http://127.0.0.1:{port}", "--model", model, "--device", "cpu", "--prepared", out,
            stderr=asyncio.subprocess.PIPE)
        _, err = await asyncio.wait_for(p.communicate(), timeout=30)
        assert p.returncode == 0, err.decode()
    finally:
        await runner.cleanup()
    return fake


def test_end_to_end_cpu(prepared):
    out, model, exe = prepared
    fake = asyncio.run(_scenario(out, model, exe))
    assert fake.events == ["ws", "ws", "ready"]
    pairs = sorted((s["story_id"], s["ticker"]) for s in fake.signals)
    assert pairs == [("s1", "AAA"), ("s1", "BBB"), ("s2", "CCC"), ("s3", "DDD")]
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForSequenceClassification.from_pretrained(model, torch_dtype=torch.float16).float().eval()
    heads = {"s1": "up up", "s2": "down down", "s3": "flat flat"}
    dirs = {0: 1, 1: -1, 2: 0}
    for s in fake.signals:
        enc = tok(heads[s["story_id"]], return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        with torch.no_grad():
            pr = torch.softmax(m(**enc).logits, -1)[0]
        assert s["direction"] == dirs[int(pr.argmax())]
        assert abs(s["confidence"] - float(pr.max())) < 1e-4


def test_tune_fallback(tmp_path):
    """tune_native never raises and leaves no arms/CSV behind when the binary is missing or fails."""
    (tmp_path / "native").mkdir()
    (tmp_path / "native" / "arms.json").write_text("garbage")
    msg = native.tune_native(tmp_path / "native" / "nope", tmp_path)
    assert "arm A" in msg
    assert not (tmp_path / "native" / "arms.json").exists()
    assert not (tmp_path / "native" / "tunableop_results.csv").exists()
