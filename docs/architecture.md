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

## Planned retrieval experiments

1. Dense retrieval baseline.
2. BM25 baseline.
3. Dense + BM25 using Reciprocal Rank Fusion.
4. Hybrid retrieval followed by cross-encoder reranking.
5. Recursive chunking compared with structure-aware chunking.

Every comparison will use the same labeled queries and corpus snapshot.
