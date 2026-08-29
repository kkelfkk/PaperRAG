# Architecture

This document records design decisions and changes as PaperRAG evolves. It is
deliberately short at project initialization; implementation details must be
supported by code and experiments before being described as completed work.

## Initial vertical slice

The first end-to-end slice will process one PDF, preserve its paper/section/page
metadata, create retrievable chunks, return the most relevant chunks for a
question, and expose the source metadata needed for citation.

## Decision 001: start with an inspectable PDF parser

The first ingestion implementation uses `pdfplumber` and stores one record per
physical PDF page. This makes page citations deterministic and keeps the first
milestone small enough to test thoroughly. A transparent heading heuristic
identifies common academic section titles and numbered headings.

This baseline intentionally does not perform OCR or reconstruct complex tables.
Docling remains a candidate for the later structure-aware ingestion experiment;
it should replace the parser through the ingestion interface rather than force
changes in retrieval code.

## Decision 002: page-aware recursive chunking

The baseline chunker splits within a physical PDF page and never combines text
from different pages. It first groups lines under detected section headings,
then recursively falls back from paragraphs to lines, words, and finally
characters. Adjacent chunks within the same section receive bounded overlap.

Every chunk stores a deterministic ID, document ID, source filename, title,
physical page number, section heading, global chunk index, and size statistics.
The conservative page boundary may sacrifice some context around page breaks,
but it makes citations unambiguous. Cross-page strategies can be evaluated later
instead of silently changing citation semantics.

## Decision 003: embedded Qdrant dense baseline

The first retriever uses Qdrant local mode with cosine similarity, allowing the
same Qdrant data model to run without Docker during early development. Each
point stores the chunk vector plus citation payload. Re-indexing a document
deletes its previous points first, preventing stale chunks after PDF updates.

Embedding generation is behind a small provider interface. The runnable
baseline uses FastEmbed's multilingual MiniLM ONNX model for modest local
resource usage; tests use deterministic vectors and require no model download.
BGE-M3 remains a later experimental candidate rather than an unmeasured default.
Collections reject attempts to mix vectors from different embedding models,
even when their dimensions happen to match.

## Decision 004: validated DeepSeek grounded generation

The first generator uses DeepSeek's Chat Completions endpoint with
`deepseek-v4-flash` in non-thinking mode. The API key is read only from the
ignored local `.env` file. Retrieved passages are labeled `[S1]`, `[S2]`, and so
on, and document text is explicitly treated as untrusted data.

The model must return a JSON object containing the answer, cited source IDs, and
an abstention flag. Application code verifies that every cited ID exists, that
the JSON list exactly matches markers in the answer, and that a non-abstained
answer cites at least one source. Empty retrieval results abstain before any API
request is made.

## Decision 005: lazy FastAPI application service

FastAPI exposes health, PDF indexing, selectable retrieval, and grounded-answer
routes.
The heavyweight embedding model and local Qdrant client are initialized only
when a pipeline endpoint is first called, then shared across requests and closed
during application lifespan shutdown. Local-mode operations are serialized to
avoid concurrent access to the embedded database.

PDF uploads are streamed in bounded blocks to a temporary file, limited to 50
MB, validated by the existing parser, and deleted in a `finally` block. Original
filenames are reduced to their basename before entering citation metadata.

## Decision 006: deterministic retrieval evaluation before optimization

Retrieval experiments use a versioned JSON dataset with a fixed `corpus_id`,
manually reviewed questions, relevant chunk IDs, and optional graded relevance.
The evaluator records the collection and embedding model, emits per-query
rankings, and computes Precision@K, Recall@K, MRR@K, and nDCG@K with macro
averages. This separates deterministic retrieval measurement from later
LLM-based generation evaluation.

The repository includes only a labeling template until a real paper corpus has
been indexed and audited. Placeholder examples must not be reported as project
results. All retrieval variants will run against the same corpus snapshot,
chunk IDs, query set, and cutoffs.

## Decision 007: BM25 plus Reciprocal Rank Fusion baseline

The lexical retriever computes Okapi BM25 directly from the chunk payloads in
Qdrant, so the baseline requires no second database or hidden preprocessing.
Latin technical terms are case-folded while contiguous Chinese text is split
into character bigrams. Titles receive a small, explicit boost by appearing
twice in the searchable text.

Hybrid retrieval requests a broader candidate list from both dense and BM25
retrievers, then combines ranks with Reciprocal Rank Fusion using equal weights
and `k=60`. RRF is used because cosine similarity and BM25 scores are not on a
shared scale. Search, answer generation, and evaluation expose `dense`, `bm25`,
and `hybrid` strategies; hybrid is the product default, while evaluation keeps
dense explicit as the original baseline.

This first BM25 implementation rebuilds corpus statistics from Qdrant payloads
for every query. That is inspectable and adequate for a small paper collection;
a persisted sparse index should replace it if profiling shows corpus scans are
the bottleneck.

## Decision 008: optional multilingual cross-encoder reranking

The `hybrid_rerank` strategy asks Hybrid Search for at least 20 candidates, then
scores every query-passage pair jointly and sorts by the new relevance score.
Candidate text includes title and section metadata as well as the chunk body.
The default model is `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, selected as a
smaller multilingual baseline; larger BGE rerankers remain experiment
candidates rather than unmeasured defaults.

The sentence-transformers adapter loads both its dependency and model lazily.
PyTorch is isolated in the optional `rerank` dependency group, and ordinary
Dense, BM25, and Hybrid use does not require it. The API therefore keeps
`hybrid` as its lightweight default and makes reranking an explicit strategy.
Tests inject a deterministic scoring provider and never download a model.

## Decision 009: one metadata filter contract for all retrieval stages

`SearchFilters` validates optional document ID, exact section, and inclusive
physical page bounds, then builds one Qdrant filter shared by Dense and BM25.
Hybrid Search and the reranker forward the same immutable object to their
candidate retrievers. Filtering therefore happens before ranking and fusion,
instead of silently removing items from an already limited Top-K.

The HTTP API, search CLI, answer CLI, and labeled evaluation format expose the
same fields. Evaluation filters are allowed only when the original question is
genuinely scoped to that document, section, or page range, preventing artificial
metric inflation.

## Decision 010: a thin Streamlit client over FastAPI

The first user interface is a separate Streamlit process that calls the public
FastAPI contract instead of importing the retrieval service directly. This
keeps model ownership, Qdrant locking, upload limits, request validation, and
secret handling in one backend process. The browser never receives the
DeepSeek API key.

The interface exposes PDF indexing, all retrieval strategies, shared metadata
filters, evidence inspection, and grounded answers with citations. Its HTTP
client converts backend and connection failures into short user-facing errors
without displaying raw transport diagnostics. Streamlit remains an optional
dependency so backend-only installation stays lightweight.

## Planned retrieval experiments

1. Dense retrieval baseline.
2. BM25 baseline.
3. Dense + BM25 using Reciprocal Rank Fusion.
4. Hybrid retrieval followed by cross-encoder reranking.
5. Recursive chunking compared with structure-aware chunking.

Every comparison will use the same labeled queries and corpus snapshot.
