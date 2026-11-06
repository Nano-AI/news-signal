"""FinBERT headline classifier."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_DIRECTIONS = {"positive": 1, "negative": -1, "neutral": 0}


def label_direction(label: str) -> int:
    try:
        return _DIRECTIONS[label.lower()]
    except KeyError:
        raise ValueError(f"unknown sentiment label: {label!r}") from None


class Classifier:
    def __init__(self, model: str, device: str):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
        self.model.eval()

    def predict(self, headline: str) -> tuple[int, float]:
        enc = self.tokenizer(headline, return_tensors="pt", truncation=True, max_length=64)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs))
        return label_direction(self.model.config.id2label[idx]), float(probs[idx])

    def predict_batch(self, headlines: list[str]) -> list[tuple[int, float]]:
        enc = self.tokenizer(headlines, return_tensors="pt", padding=True, truncation=True, max_length=64)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            probs = torch.softmax(self.model(**enc).logits, dim=-1)
        conf, idx = probs.max(dim=-1)
        id2label = self.model.config.id2label
        return [(label_direction(id2label[i]), c) for i, c in zip(idx.tolist(), conf.tolist())]

    def warmup(self) -> None:
        words = ("stocks rally as earnings beat expectations and guidance rises " * 20).split()
        for n in (1, 2, 8, 32, 64):
            for length in (4, 30, 100):
                self.predict_batch([" ".join(words[:length])] * n)
