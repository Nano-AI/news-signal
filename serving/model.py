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
