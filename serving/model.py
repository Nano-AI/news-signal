"""FinBERT headline classifier."""

import torch
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


class Classifier:
    def __init__(self, model: str, device: str):
        self.device = device
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

    def predict(self, headline: str) -> tuple[int, float]:
        enc = self.tokenizer(headline, return_tensors="pt", truncation=True, max_length=64)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs))
        return label_direction(self.model.config.id2label[idx]), float(probs[idx])

    def _capture(self) -> None:
        self._pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        for b in BATCH_BUCKETS:
            for seq in SEQ_BUCKETS:
                bufs = {k: torch.zeros(b, seq, dtype=torch.long, device=self.device)
                        for k in ("input_ids", "attention_mask", "token_type_ids")}
                bufs["input_ids"].fill_(self.tokenizer.pad_token_id)
                bufs["attention_mask"][:, 0] = 1
                with torch.inference_mode():
                    side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            torch.softmax(self.model(**bufs).logits.float(), -1)
                    torch.cuda.current_stream().wait_stream(side)
                    torch.cuda.synchronize()
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, pool=self._pool):
                        out = torch.softmax(self.model(**bufs).logits.float(), -1)
                self._graphs[(b, seq)] = (g, bufs, out)
        torch.cuda.synchronize()

    def _predict_graph(self, enc) -> torch.Tensor:
        n, length = enc["input_ids"].shape
        b, seq = pick_bucket(n, length)
        g, bufs, out = self._graphs[(b, seq)]
        padded = pad_to_bucket(enc, b, seq, self.tokenizer.pad_token_id)
        for k, v in padded.items():
            bufs[k].copy_(v)
        g.replay()
        return out[:n].cpu()

    def predict_batch(self, headlines: list[str]) -> list[tuple[int, float]]:
        enc = self.tokenizer(headlines, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
        if self.device == "cuda":
            probs = self._predict_graph(enc)
            conf, idx = probs.max(dim=-1)
            id2label = self.model.config.id2label
            return [(label_direction(id2label[i]), c) for i, c in zip(idx.tolist(), conf.tolist())]
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        id2label = self.model.config.id2label
        return [(label_direction(id2label[i]), c) for i, c in zip(idx.tolist(), conf.tolist())]

    def warmup(self) -> None:
        if self.device == "cuda":
            self._capture()
            return
        words = ("stocks rally as earnings beat expectations and guidance rises " * 20).split()
        for n in (1, 2, 8, 32, 64):
            for length in (4, 30, 100):
                self.predict_batch([" ".join(words[:length])] * n)
