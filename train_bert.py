# -*- coding: utf-8 -*-
"""
Fine-tune 中文 BERT 做「法律問題/判決書問題/混合問題」三分類。

用法:
    python train_bert.py                     # 用預設模型
    MODEL_NAME=hfl/chinese-roberta-wwm-ext python train_bert.py   # 換完整版模型

預設模型 hfl/rbt3 是 3 層的中文 RoBERTa，CPU 就能在幾分鐘內訓練完，
效果通常已足夠這種句子分類任務；如果你有 GPU，
可以換成 hfl/chinese-roberta-wwm-ext（12層完整版）追求更高上限。

輸出:
    bert_classifier/   訓練好的模型（推論時用 predict.py 載入）
"""

import csv
import os
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = os.environ.get("MODEL_NAME", "hfl/rbt3")
OUTPUT_DIR = "bert_classifier"
LABELS = ["LEGAL", "JUDGMENT", "MIXED"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 3
LR = 3e-5
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class QueryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=MAX_LEN, return_tensors="pt",
        )
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": self.enc["input_ids"][i],
            "attention_mask": self.enc["attention_mask"][i],
            "labels": self.labels[i],
        }


def load_data(path="data/dataset.csv"):
    texts, labels = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(LABEL2ID[row["label"]])
    return texts, labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"model: {MODEL_NAME} | device: {device}")

    texts, labels = load_data()
    tr_x, te_x, tr_y, te_y = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=SEED
    )
    print(f"train: {len(tr_x)}  test: {len(te_x)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label={i: l for l, i in LABEL2ID.items()},
        label2id=LABEL2ID,
    ).to(device)

    train_loader = DataLoader(
        QueryDataset(tr_x, tr_y, tokenizer), batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        QueryDataset(te_x, te_y, tokenizer), batch_size=BATCH_SIZE
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += out.loss.item()
        print(f"epoch {epoch + 1}/{EPOCHS}  loss: {total_loss / len(train_loader):.4f}")

    # ---- 評估 ----
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            preds.extend(logits.argmax(dim=-1).cpu().tolist())

    print(f"\ntest accuracy: {accuracy_score(te_y, preds):.4f}\n")
    print(classification_report(te_y, preds, target_names=LABELS, digits=4))
    print("confusion matrix (rows=實際, cols=預測, 順序 LEGAL/JUDGMENT/MIXED):")
    print(confusion_matrix(te_y, preds))

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n模型已存到 ./{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
