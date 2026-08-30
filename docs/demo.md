# Five-minute demo

This walkthrough demonstrates the implemented system without changing the
frozen evaluation data.

## One-time setup

```bash
uv sync --extra dev --extra ui --extra rerank --python 3.12
cp .env.example .env
uv run python -m scripts.eval_corpus download
uv run python -m scripts.eval_corpus index \
  --db-path storage/qdrant_eval \
  --collection paperrag_eval_v1 --recreate
```

Add a newly created DeepSeek key to `.env`. The four benchmark PDFs are public,
but answer generation still sends retrieved passages to the configured API.

For the web demo, set these values in `.env` so FastAPI opens the evaluation
corpus:

```dotenv
QDRANT_PATH=storage/qdrant_eval
QDRANT_COLLECTION=paperrag_eval_v1
PAPERRAG_CORPUS_MANIFEST=configs/eval_corpus.json
```

## Start the application

Terminal 1:

```bash
uv run uvicorn app.api.main:app --reload
```

Terminal 2:

```bash
uv run streamlit run frontend/app.py
```

Open `http://localhost:8501` and click **检查 API**.

## Demo sequence

1. Select **Hybrid + Reranker** and **只查看检索证据**.
2. Ask: `RAG-Sequence 和 RAG-Token 在使用检索文档时有什么区别？`
3. Expand the first passages and point out paper title, physical page, section,
   score, and stable chunk ID.
4. Select **多论文分解 + Hybrid + Reranker（跨论文推荐）**.
5. Ask: `比较 RAG、Self-RAG 和 CRAG 如何控制生成答案的可靠性。`
6. Show that evidence comes from multiple document IDs rather than one paper
   monopolizing Top-K.
7. Select **生成带引用回答** and rerun one question. Open the returned source
   list and match `[S1]` markers to page metadata.
8. Ask the out-of-scope question: `这些论文推荐哪种药物治疗阿尔茨海默病？`
   Explain that a safe system should abstain instead of using outside knowledge.

## What to explain while presenting

- Hybrid retrieval improves candidate diversity; the cross-encoder performs a
  slower, higher-quality second ranking stage.
- Query decomposition is deterministic and document-aware, so it adds no LLM
  token cost.
- Page-aware chunks make citations inspectable.
- Every reported metric comes from a frozen corpus and committed labels.
- The low strict citation recall is a known measured limitation, not a hidden
  failure.

## Fast fallback

If the DeepSeek balance is unavailable, use **只查看检索证据**. Retrieval,
filters, reranking, decomposition, page metadata, and source inspection all run
locally. The committed experiment documents still demonstrate the generation
results from the completed run.
