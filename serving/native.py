"""Build, export and verify the native (libtorch) runtime; see serving/native/main.cpp."""

import gzip
import json
import os
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

from serving.model import MAX_LENGTH, label_direction

DEFAULT_MODEL = "ProsusAI/finbert"
BINARY = "serving-native"
_SRC = Path(__file__).parent / "native" / "main.cpp"
_CJK = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F),
        (0x2B820, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F))


def native_dir(out) -> Path:
    return Path(out) / "native"


def build(out) -> Path:
    import torch
    from torch.utils.cpp_extension import ROCM_HOME

    nd = native_dir(out)
    nd.mkdir(parents=True, exist_ok=True)
    tdir = Path(torch.__file__).parent
    rocm = ROCM_HOME or str(Path(shutil.which("hipcc")).resolve().parent.parent)
    exe = nd / BINARY
    cmd = [
        "g++", "-std=c++17", "-O2", "-Wno-unused-result", "-Wno-deprecated-declarations",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}", "-DUSE_ROCM", "-D__HIP_PLATFORM_AMD__",
        f"-I{tdir}/include", f"-I{tdir}/include/torch/csrc/api/include", f"-I{rocm}/include",
        str(_SRC), "-o", str(exe),
        f"-L{tdir}/lib", "-Wl,--no-as-needed", "-ltorch", "-ltorch_cpu", "-ltorch_hip", "-lc10", "-lc10_hip",
        f"-L{rocm}/lib", "-lamdhip64", f"-Wl,-rpath,{tdir}/lib:{rocm}/lib",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError("native build failed: " + r.stderr[-2000:])
    return exe


def _unicode_table(path: Path) -> None:
    """Per-code-point class and NFD/strip-accent/lowercase map for cp >= 128 (derived from unicodedata only)."""
    def cls(c):
        ch = chr(c)
        cat = unicodedata.category(ch)
        if ch.isspace() or cat == "Zs":
            return 1
        if cat[0] == "C" and cat != "Cn":  # unassigned code points are ordinary (UNK) characters
            return 3
        if any(lo <= c <= hi for lo, hi in _CJK):
            return 4
        return 2 if cat[0] == "P" else 0

    lines, run = [], None
    for c in range(128, 0x110000):
        if 0xD800 <= c < 0xE000:
            k = 3
        else:
            k = cls(c)
        if run and run[2] == k and run[1] == c - 1:
            run[1] = c
        else:
            if run and run[2]:
                lines.append(f"R {run[0]} {run[1]} {run[2]}")
            run = [c, c, k]
        if k != 3 and not (0xD800 <= c < 0xE000):
            ch = chr(c)
            s = "".join(x for x in unicodedata.normalize("NFD", ch) if unicodedata.category(x) != "Mn").lower()
            if s != ch:
                lines.append(f"M {c} {len(s)} " + " ".join(str(ord(x)) for x in s))
    if run and run[2]:
        lines.append(f"R {run[0]} {run[1]} {run[2]}")
    path.write_text("\n".join(lines) + "\n")


def export_model(model: str, out) -> None:
    """Weights (fp16, packed like FusedBert), vocab, unicode table and meta for the binary."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from serving.model import FusedBert

    nd = native_dir(out)
    nd.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    m = AutoModelForSequenceClassification.from_pretrained(model, torch_dtype=torch.float16).eval()
    f = FusedBert(m)
    ws = [f.word, f.pos, f.tok[0], *f.emb_ln]
    for layer in f.layers:
        ws.extend(layer)
    ws.extend([*f.pool, *f.cls])
    torch.save([w.detach().contiguous().cpu() for w in ws], nd / "weights.pt")
    vocab = tokenizer.get_vocab()
    inv = {i: t for t, i in vocab.items()}
    (nd / "vocab.txt").write_text("\n".join(inv[i] for i in range(len(inv))) + "\n", encoding="utf-8")
    cfg = m.config
    dirs = [label_direction(cfg.id2label[i]) for i in range(3)]
    (nd / "meta.txt").write_text(
        f"heads {f.heads}\neps {f.eps!r}\ndirs {dirs[0]} {dirs[1]} {dirs[2]}\n"
        f"cls {tokenizer.cls_token_id}\nsep {tokenizer.sep_token_id}\nunk {tokenizer.unk_token_id}\n"
        f"pad {tokenizer.pad_token_id}\n")
    _unicode_table(nd / "unicode.tab")


_SYNTH = [
    "Stocks rally as earnings beat expectations", "Café Münster reports Q3 – revenue up 5% (EBITDA $1.2B)",
    "日本株が上昇 Toyota 豊田 shares", "İstanbul ǅ ß ΣΑΣ ﬁnal ﬀ", "tab\tsep\nline nbsp em", "Naïve résumé façade",
    "", "a" * 150, "supercalifragilisticexpialidocious-" * 5, "AT&T's “quoted” ‘text’ …", "x " * 120,
    "\x00ctrl\x07chars​zero﻿width", "emoji 😀 rocket 🚀 up", "ＦＵＬＬ ｗｉｄｔｈ １２３", "Ångström ñ ü ø đ ł",
]


def tokenizer_parity(model: str, out, headlines) -> tuple[int, int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    texts = list(headlines)
    ref = tokenizer(texts, truncation=True, max_length=MAX_LENGTH)["input_ids"] if texts else []
    exe = native_dir(out) / BINARY
    payload = "".join(json.dumps(t) + "\n" for t in texts)
    r = subprocess.run([str(exe), "--tokenize-stdin", "--prepared", str(out)], input=payload.encode(),
                       capture_output=True, check=True)
    got = r.stdout.decode().split("\n")[: len(texts)]
    ok = sum(1 for g, e in zip(got, ref) if g == " ".join(map(str, e)))
    return ok, len(texts)


def read_headlines(train) -> list[str]:
    hs = []
    with gzip.open(train, "rt", encoding="utf-8") as f:
        for line in f:
            hs.append(json.loads(line)["headline"])
    return hs


def write_manifest(out, enabled: bool, reason: str, model: str = DEFAULT_MODEL) -> None:
    nd = native_dir(out)
    nd.mkdir(parents=True, exist_ok=True)
    (nd / "manifest.json").write_text(json.dumps({"enabled": enabled, "reason": reason, "model": model}))


def prepare_native(train, out, device: str) -> None:
    """Never raises: a failure only disables the native runtime."""
    try:
        if device != "cuda":
            write_manifest(out, False, "device is not cuda")
            return
        exe = build(out)
        print(f"native: binary built in {exe}", flush=True)
        export_model(DEFAULT_MODEL, out)
        ok, n = tokenizer_parity(DEFAULT_MODEL, out, read_headlines(train) + _SYNTH)
        print(f"native: tokenizer parity {ok}/{n} headlines ({100.0 * ok / max(n, 1):.3f}%)", flush=True)
        if ok != n:
            write_manifest(out, False, f"tokenizer parity {ok}/{n}")
            return
        r = subprocess.run([str(exe), "--selftest", "--prepared", str(out)], capture_output=True, text=True)
        if r.returncode:
            write_manifest(out, False, "selftest failed: " + r.stderr[-500:])
            return
        write_manifest(out, True, f"tokenizer parity {ok}/{n}")
    except BaseException as e:  # noqa: BLE001 - prepare must still succeed
        print(f"native: disabled ({e!r})", file=sys.stderr, flush=True)
        try:
            write_manifest(out, False, repr(e)[:500])
        except Exception:
            pass


def select_binary(prepared, model: str, device: str):
    """Path of the native binary when it may serve this run, else None. Logs the reason."""
    if device != "cuda":
        return None
    try:
        nd = native_dir(prepared)
        man = json.loads((nd / "manifest.json").read_text())
        exe = nd / BINARY
        if not man.get("enabled") or not exe.exists():
            print(f"serving: native disabled ({man.get('reason')})", file=sys.stderr, flush=True)
            return None
        if man.get("model") != model:
            export_model(model, prepared)
            ok, n = tokenizer_parity(model, prepared, _SYNTH)
            if ok != n:
                print(f"serving: native disabled (tokenizer parity {ok}/{n} for {model})", file=sys.stderr, flush=True)
                return None
            write_manifest(prepared, True, f"re-exported for {model}", model)
        return exe
    except Exception as e:  # noqa: BLE001
        print(f"serving: native unavailable ({e!r})", file=sys.stderr, flush=True)
        return None
