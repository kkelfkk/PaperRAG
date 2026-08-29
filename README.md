# PaperRAG

An evaluation-driven RAG system for academic papers with hybrid retrieval,
reranking, and verifiable citations.

> Status: PDF ingestion milestone. PaperRAG can parse text-based PDF papers into
> structured JSON while preserving page numbers, document metadata, and likely
> section headings. Retrieval and answer generation are not implemented yet.

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
Document parsing -> Structure-aware chunking -> Dense + BM25 indexes
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
- [ ] Implement recursive chunking
- [ ] Build a dense-retrieval baseline with Qdrant
- [ ] Generate answers with page-level citations
- [ ] Add a FastAPI endpoint
- [ ] Build the first labeled evaluation set
- [ ] Add BM25 and Reciprocal Rank Fusion
- [ ] Add a cross-encoder reranker
- [ ] Compare retrieval strategies with Recall@K, MRR, and nDCG
- [ ] Evaluate faithfulness, answer relevance, and citation quality
- [ ] Add a lightweight user interface

## Proposed technology stack

- Python 3.11+
- FastAPI and Pydantic
- Docling
- Qdrant
- BGE-M3 embeddings
- BGE reranker
- OpenAI or Qwen-compatible LLM APIs
- pytest and Ruff
- Streamlit

Model and framework choices are provisional until they are validated on the
project's evaluation set.

## Repository layout

```text
app/
  api/          HTTP API
  chunking/     chunking strategies
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

Place a text-based paper PDF in `data/papers/` and parse it:

```bash
uv run python -m app.ingestion.cli data/papers/example.pdf \
  --output data/papers/example.parsed.json
```

The JSON output contains a stable document ID, SHA-256 fingerprint, PDF
metadata, total page count, extracted text for every page, and detected heading
candidates. Empty pages are retained so source page numbers remain correct.

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
