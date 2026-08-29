# Architecture

This document records design decisions and changes as PaperRAG evolves. It is
deliberately short at project initialization; implementation details must be
supported by code and experiments before being described as completed work.

## Initial vertical slice

The first end-to-end slice will process one PDF, preserve its paper/section/page
metadata, create retrievable chunks, return the most relevant chunks for a
question, and expose the source metadata needed for citation.

## Planned retrieval experiments

1. Dense retrieval baseline.
2. BM25 baseline.
3. Dense + BM25 using Reciprocal Rank Fusion.
4. Hybrid retrieval followed by cross-encoder reranking.
5. Recursive chunking compared with structure-aware chunking.

Every comparison will use the same labeled queries and corpus snapshot.
