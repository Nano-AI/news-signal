"""A randomly initialised 2-layer BERT classifier whose vocabulary is the fixture's headline words."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import BertConfig, BertForSequenceClassification, PreTrainedTokenizerFast

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
FIXTURE = Path("infosys") / "fixtures" / "tiny"


def _headlines(root: Path) -> list[str]:
    out = []
    for path in sorted((root / FIXTURE).glob("stories-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            out += [json.loads(line)["headline"] for line in f if line.strip()]
    return out


def build_tiny_model(out: Path) -> Path:
    root = Path(__file__).resolve().parents[4]
    tok = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(_headlines(root), trainers.WordLevelTrainer(special_tokens=SPECIALS))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]",
        mask_token="[MASK]",
    )
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=tok.get_vocab_size(), hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=64, max_position_embeddings=128, num_labels=3,
        id2label={0: "positive", 1: "negative", 2: "neutral"},
        label2id={"positive": 0, "negative": 1, "neutral": 2},
    )
    out.mkdir(parents=True, exist_ok=True)
    BertForSequenceClassification(config).save_pretrained(out)
    fast.save_pretrained(out)
    return out
