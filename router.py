"""
把 classifier.py 接到本地的 HiRAG。

前提：你已經 `pip install -e .` 裝好 HiRAG，並且已經分別把「法規/法律知識」
和「判決書」灌進兩個不同的 working_dir（各自是一個獨立的 HiRAG 知識庫）。

如果你目前只有一個 working_dir（法規和判決書混在一起），建議先拆成兩個，
分類器的價值才出得來——不然分類完也只能查同一個庫。

用法:
    python router.py "詐欺罪的構成要件是什麼？"
    python router.py "111年度台上字第1234號判決的爭點是什麼？"
    python router.py "侵權行為的成立要件有哪些？可以附上實務判決佐證嗎？"
"""

import json

from hirag import HiRAG, QueryParam

from predict import classify_query  # 本地 BERT 分類器，不需要任何 LLM API

# ---------------------------------------------------------------------------
# 建立兩個獨立的 HiRAG 知識庫。
# working_dir 要指向你已經 insert 過資料的資料夾。
# ---------------------------------------------------------------------------

legal_rag = HiRAG(
    working_dir="./legal_kb",       # 法規 / 法律概念知識庫
    enable_hierachical_mode=True,
)

judgment_rag = HiRAG(
    working_dir="./judgment_kb",    # 判決書知識庫
    enable_hierachical_mode=True,
)

QUERY_PARAM = QueryParam(mode="hi")


def route_query(query: str) -> dict:
    """分類問題，並查詢對應的 HiRAG 知識庫，回傳分類結果與檢索內容。"""

    classification = classify_query(query)
    category = classification["category"]

    if category == "LEGAL":
        context = legal_rag.query(query, param=QUERY_PARAM)
        sources = {"legal_kb": context}

    elif category == "JUDGMENT":
        context = judgment_rag.query(query, param=QUERY_PARAM)
        sources = {"judgment_kb": context}

    else:  # MIXED
        sources = {
            "legal_kb": legal_rag.query(query, param=QUERY_PARAM),
            "judgment_kb": judgment_rag.query(query, param=QUERY_PARAM),
        }

    return {
        "classification": classification,
        "sources": sources,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('用法: python router.py "你的法律問題"')
        sys.exit(1)

    q = " ".join(sys.argv[1:])
    result = route_query(q)
    print(json.dumps(result, ensure_ascii=False, indent=2))
