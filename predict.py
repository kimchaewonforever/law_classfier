# -*- coding: utf-8 -*-
"""
載入訓練好的 BERT 分類器做推論。

用法:
    python predict.py "詐欺罪的構成要件是什麼？"

或當模組用（接到你的 HiRAG router）:
    from predict import classify_query
    result = classify_query("詐欺罪的構成要件是什麼？")
    # {"category": "LEGAL", "category_label": "法律問題", "confidence": 0.98}
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "bert_classifier"
CATEGORY_LABELS = {"LEGAL": "法律問題", "JUDGMENT": "判決書問題", "MIXED": "混合問題"}

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _model.eval()
    return _tokenizer, _model


def classify_query(query: str) -> dict:
    tokenizer, model = _load()
    inputs = tokenizer(query, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(probs.argmax())
    category = model.config.id2label[idx]
    return {
        "category": category,
        "category_label": CATEGORY_LABELS[category],
        "confidence": round(float(probs[idx]), 4),
    }


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print('用法: python predict.py "你的法律問題"')
        sys.exit(1)

    print(json.dumps(classify_query(" ".join(sys.argv[1:])), ensure_ascii=False, indent=2))
