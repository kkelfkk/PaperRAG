# Generation evaluation

PaperRAG separates human labels, saved model outputs, and evaluation reports.
This prevents a rerun of a non-deterministic LLM from changing the object being
measured while a metric or prompt is under development.

## Implemented metrics

- **Citation precision**: relevant cited chunks divided by all cited chunks.
- **Citation recall**: relevant cited chunks divided by all human-labeled
  supporting chunks.
- **Citation F1**: harmonic mean of citation precision and recall.
- **Citation validity**: cited chunks that were present in the retrieved
  context divided by all cited chunks. A fabricated citation therefore lowers
  this score even if its identifier happens to resemble a real chunk.
- **Abstention accuracy**: whether the system's abstention decision matches the
  human answerability label.

Citation precision, recall, and F1 are averaged only over answerable questions.
Abstention accuracy covers every question. An abstained output with no citation
receives full citation validity; a non-abstained output with no citation
receives zero.

## Reproducible workflow

1. Freeze the corpus and generation questions.
2. Manually label answerability and supporting chunk IDs.
3. Run one fixed retrieval and generation configuration.
4. Save its answer, abstention flag, retrieved chunk IDs, cited chunk IDs, and
   exact evidence text.
5. Run `python -m app.evaluation.generation_cli LABELS PREDICTIONS`.

The evaluator rejects missing or unexpected query IDs and mismatched dataset
versions. It makes no network calls, so the same files always produce the same
report.

`app.evaluation.generation_run_cli` automates step 3 for the local Qdrant
corpus. It records the fixed strategy, Top-K, retrieval model chain, answer
model, cited chunks, retrieved chunks, and exact evidence. This command invokes
DeepSeek and transmits the question and retrieved passages; the offline scoring
command does not.

## Limitations and next experiment

Chunk overlap can make multiple passages valid evidence for the same claim, so
labels need a consistent annotation policy. Citation correctness also does not
prove that every sentence in an answer is supported or that the answer addresses
the question.

The prediction format therefore preserves evidence text and an optional human
reference answer. The next experiment will add a versioned judge prompt for
claim-level faithfulness and answer relevance, calibrate it against a small
human-scored subset, and report judge-model/version metadata with every score.

## Experiment 001: first real generation run

Run date: 2026-08-30

### Fixed inputs

- Dataset: `data/eval/paperrag_generation_10.json`, version `0.1.0`
- Questions: six answerable and four deliberately out-of-scope
- Corpus: the same pinned four-paper, 538-chunk snapshot used for retrieval
- Retrieval: document-aware decomposition + Hybrid + cross-encoder reranking
- Context: Top-5
- Answer model: DeepSeek V4 Flash, non-thinking mode, temperature 0
- Citation validation: unknown, missing, or inconsistent source IDs rejected;
  validation-invalid JSON may receive up to two explicit repair attempts

Predictions and evidence remain local under the ignored `data/eval/results/`
directory. Only aggregate metrics are committed.

### Results

| Metric | Score |
| --- | ---: |
| Abstention accuracy | **100.00%** |
| Citation validity | **100.00%** |
| Citation precision | 27.78% |
| Citation recall | 19.44% |
| Citation F1 | 22.22% |

All four unanswerable questions were correctly rejected, and every emitted
citation pointed to a chunk that was actually supplied to the model. However,
three of six answerable questions cited no chunk in the strict gold set. This
is not an LLM citation-format failure: it shows that Top-5 retrieval often
returns adjacent or semantically related passages rather than the exact chunks
selected during annotation.

The next controlled experiment should audit those three failures for acceptable
overlapping evidence, then compare Top-5 with a diversified Top-10 context. Gold
labels must be expanded only when a cited chunk independently supports the
answer, never merely because the current system retrieved it.
