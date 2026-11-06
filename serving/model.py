"""FinBERT headline classifier."""

import ctypes
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LENGTH = 64
BATCH_BUCKETS = (1, 2, 4, 8, 16, 32, 64)
SEQ_BUCKETS = (16, 32, 48, 64)
_DIRECTIONS = {"positive": 1, "negative": -1, "neutral": 0}


def label_direction(label: str) -> int:
    try:
        return _DIRECTIONS[label.lower()]
    except KeyError:
        raise ValueError(f"unknown sentiment label: {label!r}") from None


def pick_bucket(n: int, length: int) -> tuple[int, int]:
    """Smallest (batch, seq) bucket holding n rows of the given length."""
    b = next((x for x in BATCH_BUCKETS if x >= n), None)
    seq = next((x for x in SEQ_BUCKETS if x >= length), None)
    if b is None or seq is None:
        raise ValueError(f"no bucket for batch {n}, length {length}")
    return b, seq


def pad_to_bucket(enc, b: int, seq: int, pad_id: int) -> dict[str, torch.Tensor]:
    """Copy tokenizer output into zero-masked (b, seq) int64 tensors; pad ids are pad_id."""
    n, length = enc["input_ids"].shape
    out = {k: torch.zeros(b, seq, dtype=torch.long) for k in ("input_ids", "attention_mask", "token_type_ids")}
    out["input_ids"].fill_(pad_id)
    for k, v in out.items():
        v[:n, :length] = enc[k]
    return out


ARMS = ("fused_hip", "fused_sdpa", "fused_explicit", "hf_eager")  # candidate graphs per bucket; earlier wins ties


def load_kernels(prepared):
    """ctypes handle on prepared/bert_fused.so, or None if absent or unloadable."""
    path = Path(prepared) / "bert_fused.so" if prepared else None
    if path is None or not path.exists():
        return None
    try:
        return ctypes.CDLL(str(path))
    except OSError:
        return None


def median(xs):
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def select_arm(medians: dict[str, float]) -> str:
    """Arm with the smallest median replay time; ties go to the earliest arm in ARMS."""
    return min((a for a in ARMS if a in medians), key=lambda a: medians[a])


class FusedBert:
    """Pure-torch BERT classifier forward over weights pre-packed from an HF BertForSequenceClassification.

    Per layer: fused QKV addmm, SDPA with additive key mask, out-proj, add+LN, FFN up + GELU, FFN down, add+LN.
    """

    def __init__(self, model):
        cfg = model.config
        if cfg.hidden_act != "gelu":
            raise ValueError(f"unsupported activation {cfg.hidden_act!r}")
        self.heads = cfg.num_attention_heads
        self.hidden = cfg.hidden_size
        self.eps = cfg.layer_norm_eps
        bert = model.bert
        p = next(model.parameters())
        self.dtype = p.dtype
        e = bert.embeddings
        self.word, self.pos, self.tok = (t.weight.detach() for t in
                                         (e.word_embeddings, e.position_embeddings, e.token_type_embeddings))
        self.emb_ln = (e.LayerNorm.weight.detach(), e.LayerNorm.bias.detach())
        self.layers = []
        for l in bert.encoder.layer:
            a = l.attention.self
            qkv_w = torch.cat([a.query.weight, a.key.weight, a.value.weight], 0).detach().contiguous()
            qkv_b = torch.cat([a.query.bias, a.key.bias, a.value.bias], 0).detach().contiguous()
            ao, ln1 = l.attention.output.dense, l.attention.output.LayerNorm
            self.layers.append((
                qkv_w, qkv_b,
                ao.weight.detach(), ao.bias.detach(), ln1.weight.detach(), ln1.bias.detach(),
                l.intermediate.dense.weight.detach(), l.intermediate.dense.bias.detach(),
                l.output.dense.weight.detach(), l.output.dense.bias.detach(),
                l.output.LayerNorm.weight.detach(), l.output.LayerNorm.bias.detach()))
        self.pool = (bert.pooler.dense.weight.detach(), bert.pooler.dense.bias.detach())
        self.cls = (model.classifier.weight.detach(), model.classifier.bias.detach())

    def _attn(self, q, k, v, mask, sdpa=True):
        if sdpa:
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        w = torch.softmax(q @ k.transpose(-1, -2) * (q.shape[-1] ** -0.5) + mask, dim=-1)
        return w @ v

    def forward_hip(self, lib, input_ids, token_type_ids, attention_mask):
        """forward() structure with HIP kernels for emb+LN, attention and add+LN."""
        B, S = input_ids.shape
        H, nh = self.hidden, self.heads
        if H // nh != 64 or S > 64 or H > 1024 or self.dtype != torch.float16:
            raise RuntimeError("fused_hip unsupported for this shape")
        st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        P = lambda t: ctypes.c_void_p(t.data_ptr())
        eps, scale = ctypes.c_float(self.eps), ctypes.c_float((H // nh) ** -0.5)
        x = torch.empty(B * S, H, dtype=self.dtype, device=input_ids.device)
        lib.launch_emb_ln(P(input_ids), P(token_type_ids), P(self.word), P(self.pos), P(self.tok),
                          P(self.emb_ln[0]), P(self.emb_ln[1]), P(x), B * S, S, H, eps, st)
        for (qw, qb, ow, ob, l1w, l1b, iw, ib, dw, db, l2w, l2b) in self.layers:
            qkv = torch.addmm(qb, x, qw.t())
            a = torch.empty(B * S, H, dtype=self.dtype, device=x.device)
            lib.launch_attn(P(qkv), P(attention_mask), P(a), B, S, nh, scale, st)
            h = torch.addmm(ob, a, ow.t())
            lib.launch_add_ln(P(x), P(h), P(l1w), P(l1b), P(x), B * S, H, eps, st)
            f = F.gelu(torch.addmm(ib, x, iw.t()))
            h = torch.addmm(db, f, dw.t())
            lib.launch_add_ln(P(x), P(h), P(l2w), P(l2b), P(x), B * S, H, eps, st)
        pooled = torch.tanh(torch.addmm(self.pool[1], x.view(B, S, H)[:, 0], self.pool[0].t()))
        logits = torch.addmm(self.cls[1], pooled, self.cls[0].t())
        return torch.softmax(logits.float(), dim=-1)

    def forward(self, input_ids, token_type_ids, attention_mask, sdpa=True):
        B, S = input_ids.shape
        H, nh = self.hidden, self.heads
        d = H // nh
        x = self.word[input_ids] + self.pos[:S] + self.tok[token_type_ids]
        x = F.layer_norm(x, (H,), *self.emb_ln, self.eps)
        mask = ((1 - attention_mask.to(self.dtype)) * -1e4).view(B, 1, 1, S)
        for (qw, qb, ow, ob, l1w, l1b, iw, ib, dw, db, l2w, l2b) in self.layers:
            qkv = torch.addmm(qb, x.view(B * S, H), qw.t()).view(B, S, 3, nh, d)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            a = self._attn(q, k, v, mask, sdpa).transpose(1, 2).reshape(B * S, H)
            h = torch.addmm(ob, a, ow.t()).view(B, S, H)
            x = F.layer_norm(x + h, (H,), l1w, l1b, self.eps)
            f = F.gelu(torch.addmm(ib, x.view(B * S, H), iw.t()))
            h = torch.addmm(db, f, dw.t()).view(B, S, H)
            x = F.layer_norm(x + h, (H,), l2w, l2b, self.eps)
        pooled = torch.tanh(torch.addmm(self.pool[1], x[:, 0], self.pool[0].t()))
        logits = torch.addmm(self.cls[1], pooled, self.cls[0].t())
        return torch.softmax(logits.float(), dim=-1)


class Classifier:
    def __init__(self, model: str, device: str, prepared=None):
        self.device = device
        self.lib = load_kernels(prepared) if device == "cuda" else None
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if device == "cuda":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model, torch_dtype=torch.float16, attn_implementation="eager"
            ).to(device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
        self.model.eval()
        self._graphs: dict[tuple[int, int], tuple] = {}
        self._pool = None
        self.fused = FusedBert(self.model) if device == "cuda" else None
        self._np: dict = {}
        self._pad_id = self.tokenizer.pad_token_id
        self._id_dir = {int(i): label_direction(l) for i, l in self.model.config.id2label.items()}

    def predict(self, headline: str) -> tuple[int, float]:
        enc = self.tokenizer(headline, return_tensors="pt", truncation=True, max_length=64)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs))
        return label_direction(self.model.config.id2label[idx]), float(probs[idx])

    def _arm_fwd(self, arm, dev):
        if arm == "hf_eager":
            return lambda: torch.softmax(
                self.model(input_ids=dev[0], token_type_ids=dev[1], attention_mask=dev[2]).logits.float(), -1)
        if arm == "fused_hip":
            return lambda: self.fused.forward_hip(self.lib, dev[0], dev[1], dev[2])
        sdpa = arm == "fused_sdpa"
        return lambda: self.fused.forward(dev[0], dev[1], dev[2], sdpa)

    def _capture_arm(self, arm, dev, side):
        fwd = self._arm_fwd(arm, dev)
        with torch.inference_mode():
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    fwd()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self._pool):
                out = fwd()
        return g, out

    @staticmethod
    def _time_replay(g, warm=5, n=15) -> float:
        for _ in range(warm):
            g.replay()
        times = []
        for _ in range(n):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            g.replay()
            e1.record()
            e1.synchronize()
            times.append(e0.elapsed_time(e1))
        return median(times)

    def _capture(self) -> None:
        self._pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        pad = self.tokenizer.pad_token_id
        arms = list(ARMS)
        if self.lib is None:
            arms.remove("fused_hip")
        try:  # SDPA probe: if it raises, that arm is unavailable
            with torch.inference_mode():
                z = torch.zeros(1, 16, dtype=torch.long, device=self.device)
                self.fused.forward(z, z, torch.ones_like(z), True)
                torch.cuda.synchronize()
        except RuntimeError:
            arms.remove("fused_sdpa")
        chosen = {}
        for b in BATCH_BUCKETS:
            for seq in SEQ_BUCKETS:
                dev = torch.zeros(3, b, seq, dtype=torch.long, device=self.device)
                dev[0].fill_(pad)
                dev[2][:, 0] = 1
                host = torch.zeros(3, b, seq, dtype=torch.long).pin_memory()
                cands, meds = {}, {}
                for arm in arms:
                    try:
                        g, out = self._capture_arm(arm, dev, side)
                    except RuntimeError:
                        continue
                    cands[arm] = (g, out)
                    meds[arm] = self._time_replay(g)
                best = select_arm(meds)
                g, out = cands[best]
                del cands
                out_host = torch.empty(out.shape, dtype=out.dtype).pin_memory()
                self._graphs[(b, seq)] = (g, dev, host, out, out_host)
                chosen[(b, seq)] = best
        torch.cuda.synchronize()
        short = {"fused_hip": "K", "fused_sdpa": "S", "fused_explicit": "E", "hf_eager": "H"}
        print("graph arms (K=hip S=sdpa E=explicit H=hf): " +
              " ".join(f"{b}x{s}:{short[a]}" for (b, s), a in chosen.items()), file=sys.stderr, flush=True)

    def tokenize(self, headline: str) -> list[int]:
        """Token ids (with special tokens, truncated, unpadded) for one headline."""
        return self.tokenizer(headline, truncation=True, max_length=MAX_LENGTH)["input_ids"]

    def _host_np(self, key, host):
        v = self._np.get(key)
        if v is None:
            v = self._np[key] = host.numpy()
        return v

    def _predict_graph_ids(self, ids_list) -> list[tuple[int, float]]:
        n = len(ids_list)
        b, seq = pick_bucket(n, max(map(len, ids_list)))
        g, dev, host, out, out_host = self._graphs[(b, seq)]
        h = self._host_np((b, seq), host)
        h[0].fill(self._pad_id)
        h[1:] = 0
        ids, mask = h[0], h[2]
        for i, x in enumerate(ids_list):
            m = len(x)
            ids[i, :m] = x
            mask[i, :m] = 1
        dev.copy_(host, non_blocking=True)
        g.replay()
        out_host.copy_(out, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        probs = self._host_np(("out", b, seq), out_host)[:n]
        idx = probs.argmax(axis=-1)
        conf = probs[np.arange(n), idx].tolist()
        dirs = self._id_dir
        return [(dirs[i], c) for i, c in zip(idx.tolist(), conf)]

    def predict_ids(self, ids_list: list[list[int]]) -> list[tuple[int, float]]:
        if self.device == "cuda":
            return self._predict_graph_ids(ids_list)
        n, length = len(ids_list), max(map(len, ids_list))
        ids = torch.full((n, length), self._pad_id, dtype=torch.long)
        mask = torch.zeros(n, length, dtype=torch.long)
        for i, x in enumerate(ids_list):
            ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
            mask[i, : len(x)] = 1
        enc = {"input_ids": ids, "attention_mask": mask, "token_type_ids": torch.zeros_like(ids)}
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        return [(self._id_dir[i], c) for i, c in zip(idx.tolist(), conf.tolist())]

    def predict_batch(self, headlines: list[str]) -> list[tuple[int, float]]:
        return self.predict_ids([self.tokenize(h) for h in headlines])

    def warmup(self) -> None:
        if self.device == "cuda":
            self._capture()
            return
        words = ("stocks rally as earnings beat expectations and guidance rises " * 20).split()
        for n in (1, 2, 8, 32, 64):
            for length in (4, 30, 100):
                self.predict_batch([" ".join(words[:length])] * n)
