# Retrieval evaluation

## Experiment 001: first manually reviewed subset

Run date: 2026-08-29

This experiment answers a narrow question: how do the four implemented
retrieval strategies behave when the query is Chinese and the indexed academic
paper is English?

### Fixed inputs

- Dataset: `data/eval/paperrag_retrieval_10.json`, version `0.1.0`
- Questions: 10 total, covering five RAG and five ReAct questions
- Corpus: four version-pinned papers, 538 chunks
- Corpus fingerprint: `eccd943ec6d334ad77fa2478e785fc71c77338e2ebd78fa1a93324aea7144eda`
- Chunking: maximum 1,200 characters with 200-character overlap
- Dense model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Sparse retrieval: BM25
- Fusion: Reciprocal Rank Fusion
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- Cutoffs: 1, 3, 5, and 10; no metadata filters

Every relevance label was checked against the PDF text and page number. The
retriever output itself was never used as ground truth. Notes beside each query
record why its chunks were marked relevant.

### Results

| Strategy | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 10.00% | 10.00% | 8.33% | 6.19% |
| BM25 | 18.33% | 30.83% | 32.25% | 28.45% |
| Hybrid RRF | 10.83% | 22.50% | 16.17% | 15.18% |
| Hybrid + reranker | **34.17%** | **45.83%** | **35.00%** | **32.58%** |

### Interpretation

The multilingual cross-encoder gives the best result on every reported macro
metric and raises Recall@10 by 35.83 percentage points over dense retrieval.
This shows the value of scoring the broad Hybrid candidate set a second time.

BM25 is stronger than the dense baseline despite the language mismatch. Many
queries retain English technical names such as `RAG-Sequence`, `HotpotQA`, and
`in-context examples`, which provide useful exact lexical matches. The current
MiniLM dense embedding does not reliably align the surrounding Chinese wording
with English paper passages.

Plain RRF fusion does not automatically beat its strongest component. Weak
dense rankings can pull strong BM25 hits downward, so fusion weights and the
candidate pool need evaluation instead of being chosen by intuition.

### Limitations and next experiment

Ten questions are too few for a final project claim, and the set currently
covers only RAG and ReAct. Comparison and multi-paper questions are also likely
to be harder. These numbers must be presented as development-set results.

The next experiment will label the 10 Self-RAG and CRAG questions, then compare
the same four strategies without changing the existing labels. After all 30
questions are reviewed, tune RRF weights on a development split and report the
final result on a held-out split.
