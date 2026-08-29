# Retrieval evaluation data

`retrieval.template.json` documents the versioned format. Copy it to a new file
before labeling; do not report metrics from the placeholder template.

For each question:

1. Freeze the paper corpus and chunking configuration.
2. Search broadly and manually inspect the source pages.
3. Record every relevant `chunk_id`, not only the current top result.
4. Optionally assign positive integer grades, where larger means more relevant.
5. Add a short annotation note so another person can audit the label.

An evaluation query may include `document_id`, `section`, `page_from`, and
`page_to`. Use them only when the user-facing task itself supplies that scope;
do not add filters merely to make retrieval scores look better.

Start with 30 reviewed questions across factual, terminology, comparison, and
multi-evidence categories. Grow to 100-200 only after the annotation process is
consistent.

`question_seeds.json` contains the first 30 questions for the pinned public
corpus in `configs/eval_corpus.json`. Generate the local annotation worksheet:

```bash
uv run python -m scripts.eval_corpus download
uv run python -m scripts.eval_corpus index \
  --db-path storage/qdrant_eval --collection paperrag_eval_v1 --recreate
uv run python -m scripts.eval_corpus worksheet \
  --db-path storage/qdrant_eval --collection paperrag_eval_v1
```

The generated `data/eval/work/retrieval_candidates.json` deliberately contains
empty labels. Inspect source pages, add all relevant chunk IDs, record uncertain
decisions, and have a second pass review comparison and multi-evidence items.

Validate a dataset without loading Qdrant or an embedding model:

```bash
uv run python -m app.evaluation.cli data/eval/my_retrieval.json --validate-only
```

Run retrieval evaluation:

```bash
uv run python -m app.evaluation.cli data/eval/my_retrieval.json \
  --strategy hybrid \
  --cutoffs 1 3 5 10 \
  --output data/eval/results/hybrid-rrf.json
```

Use `--strategy dense`, `--strategy bm25`, `--strategy hybrid`, and
`--strategy hybrid_rerank` with the same dataset to make a controlled
comparison. The reranking strategy requires the `rerank` optional dependency.
