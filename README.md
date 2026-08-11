# 法律 RAG 問題分類器（BERT 版，免 LLM API）

判斷使用者問題屬於「法律問題（LEGAL）/ 判決書問題（JUDGMENT）/ 混合問題（MIXED）」，
用來決定要查法律圖譜、判決書圖譜，還是兩個都查。訓練好之後推論完全在本地跑，
不需要呼叫任何 LLM API。

## 檔案說明

| 檔案 | 用途 |
|---|---|
| `generate_dataset.py` | 生成訓練資料（模板×法律主題展開 + 手寫困難範例），輸出 `data/dataset.csv` |
| `data/dataset.csv` | 1545 筆已生成的標註資料（三類各 515 筆） |
| `train_bert.py` | Fine-tune 中文 BERT，訓練完存到 `bert_classifier/` |
| `predict.py` | 載入訓練好的模型做推論（完全本地，不需任何 LLM API） |
| `router.py` | 把分類結果接到兩個 HiRAG 圖譜的路由範例 |

## 使用步驟

```bash
# 1. 安裝套件（有 GPU 的話 torch 裝 CUDA 版）
pip install torch transformers scikit-learn numpy

# 2. 訓練（預設用 hfl/rbt3，3層小模型，CPU 幾分鐘可完成）
python train_bert.py

# 想用完整 12 層模型（建議有 GPU 再用）：
MODEL_NAME=hfl/chinese-roberta-wwm-ext python train_bert.py

# 3. 推論
python predict.py "詐欺罪的構成要件是什麼？"
```

接到 HiRAG 的方式：`router.py` 已直接使用 `predict.classify_query`，
根據分類結果查法律圖譜、判決書圖譜或兩者。

## 重要注意事項：測試準確度會虛高

這份資料集是模板展開生成的，訓練/測試集共用同一批模板，所以測出來的
accuracy 會接近 100%（我們用最簡單的 TF-IDF baseline 都能拿到 1.0）。
**這個數字不代表真實效果**——真實使用者的問法會比模板多變得多。

建議上線前後做兩件事：

1. **收集真實使用者問題**：上線初期把使用者實際輸入的問題記下來，
   人工標 100~200 筆當「真實測試集」，用它來評估才準。
2. **持續補資料重訓**：把分類錯誤或 confidence 低（例如 < 0.7）的真實問題
   標好類別後加進 `data/dataset.csv` 重新訓練，效果會逐步變好。
   confidence 低的問題也可以在系統裡直接 fallback 成 MIXED（兩個圖譜都查），
   寧可多查也不要漏。

## 想調整類別定義？

直接改 `generate_dataset.py` 裡的模板和手寫範例，重跑
`python generate_dataset.py && python train_bert.py` 即可。
