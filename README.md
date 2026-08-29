# PaperRAG

An evaluation-driven RAG system for academic papers with hybrid retrieval,
reranking, and verifiable citations.

> Status: evaluation-ready advanced RAG baseline. PaperRAG combines dense and
> BM25 retrieval with Reciprocal Rank Fusion, optionally applies multilingual
> cross-encoder reranking, and returns grounded DeepSeek answers with citations.

## Why PaperRAG?

General-purpose RAG demos often lose document structure, miss exact technical
terms, and return citations that are difficult to verify. PaperRAG focuses on
academic papers and will improve these weaknesses through structure-aware
chunking, hybrid retrieval, reranking, and evaluation.

## MVP scope

The first working version will:

- import local PDF papers;
- preserve paper, section, and page metadata;
- split papers into retrievable chunks;
- build a dense vector index;
- answer questions using retrieved evidence;
- return verifiable paper and page citations.

The first version will not include an autonomous research agent, automatic web
crawling, a knowledge graph, or multi-agent workflows.

## Planned architecture

```text
PDF papers
    |
    v
Document parsing -> Structure-aware chunking -> Qdrant chunk/vector index
                                                    |
User question -> Query processing -> Hybrid retrieval -> Reranking
                                                    |
                                                    v
                                      Context assembly -> LLM
                                                    |
                                                    v
                                      Answer + verified citations
```

## Roadmap

- [x] Define the project scope and repository structure
- [x] Parse a text-based PDF and preserve section/page metadata
- [x] Implement page-aware recursive chunking
- [x] Build a dense-retrieval baseline with Qdrant
- [x] Generate DeepSeek answers with validated page-level citations
- [x] Add FastAPI endpoints for indexing, search, and grounded answers
- [x] Add a versioned retrieval evaluation format, runner, and IR metrics
- [ ] Manually label the first 30-question evaluation set
- [x] Add BM25 and Reciprocal Rank Fusion
- [x] Add a cross-encoder reranker
- [ ] Compare retrieval strategies on the labeled set
- [ ] Evaluate faithfulness, answer relevance, and citation quality
- [ ] Add a lightweight user interface

## Proposed technology stack

- Python 3.11+
- FastAPI and Pydantic
- Docling
- Qdrant
- FastEmbed multilingual MiniLM baseline; BGE-M3 planned for comparison
- Multilingual MiniLM cross-encoder baseline; BGE reranker planned for comparison
- DeepSeek Chat Completions API
- pytest and Ruff
- Streamlit

Model and framework choices are provisional until they are validated on the
project's evaluation set.

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
- `POST /v1/documents/index` - upload and index a PDF (50 MB limit);
- `POST /v1/search` - return ranked evidence with source metadata;
- `POST /v1/ask` - generate a validated DeepSeek answer with citations.

The `/v1/search` and `/v1/ask` JSON bodies accept `strategy` as `dense`, `bm25`,
`hybrid`, or `hybrid_rerank`; the lightweight default remains `hybrid`.

The embedding model and database are loaded lazily, so `/health` does not trigger
a model download. Uploaded PDFs are processed through a temporary file and
deleted after indexing.

Copy the retrieval evaluation template and replace every placeholder with
manually verified chunk IDs from one fixed corpus:

```bash
cp data/eval/retrieval.template.json data/eval/my_retrieval.json
uv run python -m app.evaluation.cli \
  data/eval/my_retrieval.json --validate-only
```

After the dataset validates, evaluate all three strategies against exactly the
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
```

The report includes each query's retrieved chunk IDs and Precision@K,
Recall@K, MRR@K, and nDCG@K, followed by macro averages. The committed template
contains placeholders, not benchmark results; scores should only be published
after real papers and labels are fixed.

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

## License

This project is licensed under the MIT License.
