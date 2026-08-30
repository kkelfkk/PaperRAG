# PaperRAG

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An evaluation-driven RAG system for academic papers with hybrid retrieval,
cross-encoder reranking, multi-paper query decomposition, and verified
page-level citations.

> **Status:** complete portfolio-ready RAG baseline. The repository includes a
> FastAPI backend, Streamlit interface, frozen public-paper corpus, 30-question
> retrieval benchmark, 10-question generation benchmark, and 114 automated
> tests.

## Highlights

- Parses text-based PDFs while preserving document, section, and physical page
  metadata for auditable citations.
- Combines multilingual dense retrieval and BM25 with Reciprocal Rank Fusion,
  then optionally reranks candidates with a multilingual cross-encoder.
- Decomposes named multi-paper questions into document-filtered subqueries and
  merges balanced evidence rankings.
- Generates DeepSeek answers from retrieved evidence only, validates every
  source marker, repairs citation-format failures without weakening validation,
  and abstains when evidence is insufficient.
- Measures retrieval with Recall@K, MRR, and nDCG; measures generation with
  citation precision/recall/validity and abstention accuracy.

Measured development-set results:

- Query decomposition raised full-benchmark Recall@10 from **31.67% to 35.92%**.
- On cross-paper questions, Recall@10 rose from **11.67% to 24.42%**.
- The first real generation run achieved **100% citation validity** and **100%
  abstention accuracy**, while exposing low strict citation recall as the next
  retrieval/annotation problem to solve.

See the [retrieval experiments](docs/evaluation.md), [generation
experiments](docs/generation_evaluation.md), [five-minute demo](docs/demo.md),
and [resume/interview notes](docs/resume.md).

## Problem

General-purpose RAG demos often lose document structure, miss exact technical
terms, and return citations that are difficult to verify. PaperRAG focuses on
academic papers and makes those failure modes measurable instead of hiding them
behind a chat interface.

## Architecture

```text
PDF papers
    |
    v
Document parsing -> Structure-aware chunking -> Qdrant chunk/vector index
                                                    |
User question -> Query processing -> Hybrid retrieval -> Reranking
                         |                          |
                         +-> multi-paper split ----+
                                                    |
                                                    v
                                      Context assembly -> LLM
                                                    |
                                                    v
                                      Answer + verified citations
```

## Project status

- [x] Define the project scope and repository structure
- [x] Parse a text-based PDF and preserve section/page metadata
- [x] Implement page-aware recursive chunking
- [x] Build a dense-retrieval baseline with Qdrant
- [x] Generate DeepSeek answers with validated page-level citations
- [x] Add FastAPI endpoints for indexing, search, and grounded answers
- [x] Add a versioned retrieval evaluation format, runner, and IR metrics
- [x] Build a pinned four-paper corpus and 30-question annotation worksheet
- [x] Manually verify relevant chunks for the 30-question evaluation set
- [x] Add BM25 and Reciprocal Rank Fusion
- [x] Add a cross-encoder reranker
- [x] Add document, section, and page-range metadata filters
- [x] Compare four retrieval strategies on the first 10 labeled questions
- [x] Add query decomposition for cross-paper comparison questions
- [x] Evaluate citation precision/recall, citation validity, and abstention
- [ ] Add LLM-judged faithfulness and answer relevance
- [x] Add a lightweight Streamlit user interface

Deliberately out of scope for this baseline: autonomous web crawling, knowledge
graphs, and multi-agent research workflows. The next measured experiment is
LLM-judged faithfulness and answer relevance calibrated against human scores.

## Technology stack

- Python 3.11+
- FastAPI and Pydantic
- pdfplumber
- Qdrant
- FastEmbed multilingual MiniLM
- Multilingual MiniLM cross-encoder
- DeepSeek Chat Completions API
- pytest and Ruff
- Streamlit

The model choices are explicit baselines rather than claims that the largest
available model is automatically best.

## Repository layout

```text
app/
  api/          HTTP API
  chunking/     chunking strategies
  evaluation/   labeled datasets, retrieval metrics, and evaluation runner
  generation/   prompts, context assembly, and citations
  ingestion/    document parsing and indexing
  reranking/    second-stage ranking
  retrieval/    dense, sparse, and hybrid retrieval
configs/        non-secret project configuration
data/eval/      labeled evaluation examples
data/papers/    local papers (ignored by Git)
docs/           architecture and experiment notes
frontend/       user interface
scripts/        development and evaluation commands
tests/          automated tests
```

## Local development

Python 3.11 or 3.12 is recommended. Install the locked project and development
dependencies into an isolated environment:

```bash
uv sync --extra dev --python 3.12
```

Install the local web interface dependencies with:

```bash
uv sync --extra dev --extra ui --python 3.12
```

Install the optional local cross-encoder runtime when you are ready to test
reranking (it also installs PyTorch, so it is intentionally not a base
dependency):

```bash
uv sync --extra dev --extra rerank --python 3.12
```

Place a text-based paper PDF in `data/papers/` and parse it:

```bash
uv run python -m app.ingestion.cli data/papers/example.pdf \
  --output data/papers/example.parsed.json
```

The JSON output contains a stable document ID, SHA-256 fingerprint, PDF
metadata, total page count, extracted text for every page, and detected heading
candidates. Empty pages are retained so source page numbers remain correct.

Parse and split the PDF into retrieval-ready chunks:

```bash
uv run python -m app.chunking.cli data/papers/example.pdf \
  --output data/papers/example.chunks.json
```

The default chunk size is at most 1,200 characters with up to 200 characters of
overlap. These are explicit baseline parameters, not claimed optimal values;
they will be compared with other chunking strategies on the evaluation set.

Index one paper in an embedded local Qdrant database:

```bash
uv run python -m app.retrieval.cli index data/papers/example.pdf
```

The first indexing run downloads the default multilingual embedding model
(about 220 MB). Search the indexed chunks:

```bash
uv run python -m app.retrieval.cli search \
  "What problem does the paper solve?" --top-k 5
```

Search defaults to hybrid retrieval. Dense retrieval handles semantic matches,
BM25 handles exact technical terms, and Reciprocal Rank Fusion merges their
candidate rankings without comparing incompatible raw scores. You can inspect
each strategy separately:

```bash
uv run python -m app.retrieval.cli search "cross-encoder" --strategy dense
uv run python -m app.retrieval.cli search "cross-encoder" --strategy bm25
uv run python -m app.retrieval.cli search "cross-encoder" --strategy hybrid
uv run python -m app.retrieval.cli search \
  "cross-encoder" --strategy hybrid_rerank
```

`hybrid_rerank` retrieves at least 20 Hybrid candidates, scores every
query-passage pair jointly with the multilingual cross-encoder, and returns the
new Top-K. The first run downloads the configured reranker model. Override it
with `--reranker-model` or `RERANKER_MODEL` in `.env`.

Restrict any retrieval strategy to one paper, an exact section, and/or a
physical page range:

```bash
uv run python -m app.retrieval.cli search \
  "How is the model trained?" \
  --strategy hybrid \
  --document-id YOUR_DOCUMENT_ID \
  --section "Methods" \
  --page-from 4 --page-to 10
```

Filters are applied inside Qdrant before Dense or BM25 candidates are ranked.
They can be combined, and invalid page ranges are rejected.

The command returns ranked JSON results containing scores, chunk text, paper
title, section, and physical page number. Re-indexing the same document replaces
its previous chunks instead of creating duplicates. Local Qdrant data is written
under `storage/` and is ignored by Git.

Create a local secrets file and add a newly generated DeepSeek API key:

```bash
cp .env.example .env
```

Never commit `.env` or paste a real key into source code. After indexing at
least one paper, generate a grounded answer:

```bash
uv run python -m app.generation.cli \
  "这篇论文如何利用外部证据？" --top-k 5
```

The generator sends only the retrieved passages to DeepSeek, requires JSON
output, validates every `[S1]`-style marker, rejects unknown or missing source
IDs, and returns the corresponding paper, section, and physical page metadata.
If retrieval returns no evidence, it abstains without calling the LLM API.

Start the HTTP API:

```bash
uv run uvicorn app.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API interface. Available
endpoints are:

- `GET /health` - lightweight health check;
- `GET /v1/documents` - list indexed papers with persistent IDs and counts;
- `POST /v1/documents/index` - upload and index a PDF (50 MB limit);
- `POST /v1/search` - return ranked evidence with source metadata;
- `POST /v1/ask` - generate a validated DeepSeek answer with citations.

The `/v1/search` and `/v1/ask` JSON bodies accept `strategy` as `dense`, `bm25`,
`hybrid`, `hybrid_rerank`, or `decomposed_hybrid_rerank`; the lightweight
default remains `hybrid`. The decomposition strategy reads paper IDs and
aliases from `PAPERRAG_CORPUS_MANIFEST`.
They also accept optional `document_id`, `section`, `page_from`, and `page_to`
filters.

The embedding model and database are loaded lazily, so `/health` does not trigger
a model download. Uploaded PDFs are processed through a temporary file and
deleted after indexing.

## Streamlit interface

Run the API and interface in two terminals. Terminal 1:

```bash
uv run uvicorn app.api.main:app --reload
```

Terminal 2:

```bash
uv run streamlit run frontend/app.py
```

Open `http://localhost:8501`. The interface supports PDF upload and indexing,
an indexed-paper list and title-based scope selector, all five retrieval
strategies, document/section/page filters, evidence-only search, grounded
answers, and expandable source passages. It communicates only with the local
FastAPI service; the DeepSeek key remains in the API process's ignored `.env`
file and is never sent to the browser.

Copy the retrieval evaluation template and replace every placeholder with
manually verified chunk IDs from one fixed corpus:

```bash
cp data/eval/retrieval.template.json data/eval/my_retrieval.json
uv run python -m app.evaluation.cli \
  data/eval/my_retrieval.json --validate-only
```

PaperRAG includes a reproducible starter corpus containing version-pinned RAG,
ReAct, Self-RAG, and CRAG papers. PDF checksums are verified before indexing.
Use a separate local database so your personal knowledge base is untouched:

```bash
uv run python -m scripts.eval_corpus download
uv run python -m scripts.eval_corpus index \
  --db-path storage/qdrant_eval \
  --collection paperrag_eval_v1 --recreate
uv run python -m scripts.eval_corpus worksheet \
  --db-path storage/qdrant_eval \
  --collection paperrag_eval_v1 --top-k 20
```

The last command writes `data/eval/work/retrieval_candidates.json`, containing
30 questions and the top 20 Hybrid candidates for each. This working file is
ignored by Git. A reviewer must inspect the cited PDF pages and fill in every
`relevant_chunk_ids` list before converting it to the validated evaluation
format; empty labels are never treated as benchmark data.

After the dataset validates, evaluate all five strategies against exactly the
same labels and corpus:

```bash
uv run python -m app.evaluation.cli data/eval/my_retrieval.json \
  --strategy dense --output data/eval/results/dense-baseline.json
uv run python -m app.evaluation.cli data/eval/my_retrieval.json \
  --strategy bm25 --output data/eval/results/bm25-baseline.json
uv run python -m app.evaluation.cli data/eval/my_retrieval.json \
  --strategy hybrid \
  --cutoffs 1 3 5 10 \
  --output data/eval/results/hybrid-rrf.json
uv run python -m app.evaluation.cli data/eval/my_retrieval.json \
  --strategy hybrid_rerank \
  --cutoffs 1 3 5 10 \
  --output data/eval/results/hybrid-rerank.json
uv run python -m app.evaluation.cli data/eval/my_retrieval.json \
  --strategy decomposed_hybrid_rerank \
  --corpus-manifest configs/eval_corpus.json \
  --cutoffs 1 3 5 10 \
  --output data/eval/results/decomposed-hybrid-rerank.json
```

The report includes each query's retrieved chunk IDs and Precision@K,
Recall@K, MRR@K, and nDCG@K, followed by macro averages. The committed template
contains placeholders, not benchmark results; scores should only be published
after real papers and labels are fixed.

### Current retrieval benchmark

The completed `1.0.0` baseline contains 30 questions across RAG, ReAct,
Self-RAG, CRAG, and cross-paper comparisons. All strategies used the same
538-chunk corpus, labels, and Top-10 cutoff:

| Strategy | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 9.31% | 11.53% | 17.89% | 8.65% |
| BM25 | 7.78% | 21.39% | 17.98% | 15.36% |
| Hybrid RRF | 12.92% | 17.64% | 15.28% | 11.93% |
| Hybrid + reranker | 25.28% | 31.67% | 30.46% | 22.30% |
| Decomposition + hybrid + reranker | **26.19%** | **35.92%** | **37.59%** | **26.49%** |

The cross-encoder is the strongest conventional baseline. Deterministic query
decomposition recognizes named papers, issues one document-filtered subquery
per paper, and merges the balanced rankings with RRF. On the final cross-paper
subset it raises Recall@10 from 11.67% to 24.42%; the full benchmark rises from
31.67% to 35.92%. These are development-set results rather than a claim about
unseen questions.
See [`docs/evaluation.md`](docs/evaluation.md) for the fixed configuration,
interpretation, and limitations.

Run the tests with:

```bash
uv run pytest
```

The current parser targets text-based PDFs. Scanned documents require OCR, and
complex tables or multi-column layouts will need a layout-aware parser in a
later milestone.

## Evaluation plan

PaperRAG will use a small manually reviewed question set containing factual,
terminology, comparison, multi-evidence, and unanswerable questions.

Retrieval will be measured with Recall@5, Recall@10, MRR@10, and nDCG@10.
Generation will be evaluated for faithfulness, answer relevance, citation
precision/recall, and abstention accuracy. Experimental results will only be
published after they can be reproduced from committed configurations.

The deterministic generation evaluator is available now. Copy the two
templates in `data/eval/`, add human-reviewed labels and saved system outputs,
then run:

```bash
uv run python -m app.evaluation.generation_cli \
  data/eval/my_generation_labels.json \
  data/eval/my_generation_predictions.json \
  --output data/eval/results/generation-report.json
```

It reports chunk-level citation precision, recall, F1, citation validity
(whether every citation came from retrieved context), and abstention accuracy.
Saving evidence text beside each prediction prepares the same frozen outputs
for later faithfulness and answer-relevance judging without calling the answer
model again. See [`docs/generation_evaluation.md`](docs/generation_evaluation.md)
for definitions and limitations.

`data/eval/paperrag_generation_10.json` is the first development set: six
answerable questions inherit frozen chunk labels from the retrieval benchmark,
and four deliberately out-of-scope questions test abstention. Generate one
saved prediction file with:

```bash
uv run python -m app.evaluation.generation_run_cli \
  data/eval/paperrag_generation_10.json \
  --output data/eval/results/paperrag_generation_10_predictions.json \
  --db-path storage/qdrant_eval \
  --collection paperrag_eval_v1
```

This command makes external DeepSeek API calls and sends each question plus
its retrieved paper passages. Use it only for documents you are allowed to
send to that provider. Prediction files remain local and are ignored by Git.

### Current generation benchmark

The first 10-question development run uses DeepSeek V4 Flash, document-aware
Hybrid retrieval plus reranking, Top-5 context, six answerable questions, and
four deliberately unanswerable questions:

| Metric | Score |
| --- | ---: |
| Abstention accuracy | **100.00%** |
| Citation validity | **100.00%** |
| Citation precision | 27.78% |
| Citation recall | 19.44% |
| Citation F1 | 22.22% |

The perfect validity score means every citation referred to context that was
actually supplied; it does not mean every cited chunk matched the strict human
gold label. Three of six answerable questions cited no gold chunk, revealing a
retrieval/annotation-coverage weakness that would be hidden by reporting only
well-formed citations. These are small development-set results, not final
held-out performance.

## License

This project is licensed under the MIT License.
