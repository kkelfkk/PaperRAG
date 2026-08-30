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
