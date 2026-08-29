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

Experiment 002 therefore labels the 10 Self-RAG and CRAG questions and compares
the same four strategies without changing the existing labels.

## Experiment 002: Self-RAG and CRAG extension

Run date: 2026-08-29

The second experiment keeps the original 10 questions and labels unchanged,
then adds five manually reviewed Self-RAG questions and five manually reviewed
CRAG questions. The versioned dataset is
`data/eval/paperrag_retrieval_20.json` (`0.2.0`). Its 20 questions reference 48
unique relevant chunks, all verified to exist in the fixed corpus snapshot.

All retrieval models, fusion settings, cutoffs, and the corpus fingerprint are
identical to Experiment 001.

### Results

| Strategy | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 11.25% | 14.58% | 16.83% | 8.70% |
| BM25 | 10.42% | 29.58% | 24.47% | 21.92% |
| Hybrid RRF | 16.25% | 22.08% | 18.25% | 14.90% |
| Hybrid + reranker | **34.17%** | **41.67%** | **39.00%** | **30.31%** |

### Interpretation

The main conclusion from the first experiment remains stable after doubling
the dataset. Hybrid plus reranking ranks first on all four macro metrics. Its
Recall@10 is 27.09 percentage points above Dense and 12.09 points above BM25.

The new Self-RAG/CRAG half is not simply a copy of the earlier behavior. Dense
Recall@10 on these 10 questions is 19.17%, while BM25 reaches 28.33%, Hybrid
21.67%, and Hybrid plus reranking 37.50%. The reranker remains strongest, but
the gap and ordering of the candidate retrievers vary by paper and terminology.

Plain Hybrid now leads Dense and BM25 at Recall@5, yet BM25 is stronger at
Recall@10. This suggests RRF is improving some early ranks while still losing
useful sparse candidates deeper in the list. Candidate depth, RRF weights, and
query translation should be controlled variables in a later experiment.

### Next experiment

Manually review the final 10 cross-paper comparison questions. These require
evidence from two to four papers and should receive a second annotation pass.
Only after the 30-question set is complete should fusion weights or query
rewriting be tuned, otherwise the benchmark would leak into system design.

## Experiment 003: completed cross-paper benchmark

Run date: 2026-08-29

The final baseline adds 10 comparison and multi-evidence questions. Each label
set was checked to include evidence from every paper named in its question;
the automated coverage audit found no missing target papers. The resulting
`data/eval/paperrag_retrieval_30.json` dataset is version `1.0.0`, contains 30
questions, and references 62 unique relevant chunks.

The corpus, chunking, models, fusion configuration, and cutoffs remain unchanged
from Experiments 001 and 002.

### Results

| Strategy | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 9.31% | 11.53% | 17.89% | 8.65% |
| BM25 | 7.78% | 21.39% | 17.98% | 15.36% |
| Hybrid RRF | 12.92% | 17.64% | 15.28% | 11.93% |
| Hybrid + reranker | **25.28%** | **31.67%** | **30.46%** | **22.30%** |

### Cross-paper subset analysis

On the final 10 questions alone, Hybrid plus reranking reaches 7.50% Recall@5
and 11.67% Recall@10. Dense, BM25, and plain Hybrid reach 5.42%, 5.00%, and
8.75% Recall@10 respectively. Five of the 10 cross-paper questions retrieve no
relevant chunk in the reranked Top-10.

This is an important negative result. A single ranking list is poorly matched
to questions that require balanced evidence from two to four documents. A
high-scoring passage from one paper can occupy the candidate budget while
other required papers remain absent. Increasing Top-K alone would raise cost
without guaranteeing balanced document coverage.

### Conclusion and next hypothesis

Hybrid plus reranking is the strongest of the four implemented baselines on
all reported metrics, but reranking cannot recover evidence that never enters
its candidate set. The next hypothesis is that decomposing a comparison into
paper- or aspect-specific subqueries, retrieving for each subquery, and then
merging candidates with document-aware diversification will improve
multi-evidence Recall@10.

The `1.0.0` labels must remain frozen while testing that hypothesis. Any RRF
weight, query rewriting, or candidate-diversification tuning should use an
explicit development split, with the remaining questions held out for the
final comparison.
