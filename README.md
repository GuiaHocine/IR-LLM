# IR-LLM

Experiments connecting information retrieval and small language models: learn a sparse retriever, augment its training data with generated queries, and compare retrieval-augmented answers with and without model reasoning.

The project uses the **LoTTE technology development/search** dataset. It implements a BM25 baseline, a SPLADE-style encoder built on DistilBERT, cross-encoder distillation, and RAG with Qwen3. This is a research implementation; no benchmark results or pretrained project checkpoints are included.

## What is implemented

- **Sparse retrieval:** masked-language-model logits → `log(1 + ReLU)` → max pooling, with FLOPS sparsity regularization.
- **Training:** contrastive learning with in-batch and BM25-mined negatives, plus a separate student trained with cross-encoder ranking distillation.
- **Synthetic queries:** a small instruction model generates queries from training documents. Documents relevant to held-out queries are excluded from augmentation.
- **Evaluation:** reciprocal rank at 10 and recall at 20 on held-out queries; RAG answers saved separately for thinking and non-thinking modes.
- **Optional answer judging:** the generation model rates relevance and helpfulness. These are exploratory self-ratings, not ground-truth accuracy or factuality measurements.

## Setup

Use **Python 3.11–3.13** and **Java 17** for the recommended environment. A GPU is useful for training and generation; CPU execution is supported but can be slow. The first experiment run downloads dataset files, model checkpoints, and Terrier resources, so it requires network access and disk space.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
java -version
```

Alternatively, with `uv` already installed:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e '.[dev]'
```

For a CPU-only Torch installation, install it from the [official CPU wheel index](https://pytorch.org/get-started/locally/) before installing this project:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

Dataset and Hugging Face caches follow their normal defaults. Set `IR_DATASETS_HOME` or `HF_HOME` if you want them on another disk; the scripts do not overwrite these variables.

## Run the experiments

Every command supports `--help` without loading a model or dataset. Commands below use the same corpus limit, query limit, and seed so their splits agree.

### 1. Train and compare retrievers

```bash
ir-llm-train --epochs 1 --skip-distillation
```

Remove `--skip-distillation` to train both independent encoder variants:

```bash
ir-llm-train --epochs 3
```

The command creates or reuses a Terrier index with stored document text, mines hard negatives from training queries, and evaluates BM25 and the trained encoders on the same held-out queries and corpus. Model directories, run files, training loss histories, split queries, configuration, and retrieval metrics are written to `outputs/training/`.

### 2. Generate optional training augmentation

```bash
ir-llm-synthetic --num-docs 100 --queries-per-doc 2
ir-llm-train --synthetic-data outputs/synthetic_pairs.json --epochs 3
```

The synthetic model defaults to `HuggingFaceTB/SmolLM2-1.7B-Instruct`. Output is a JSON list:

```json
[
  {"query": "How does virtual memory work?", "doc_id": "example-document", "source": "synthetic"}
]
```

The example illustrates the schema; its document ID is not a dataset reference. A companion `.config.json` records the generation settings and split IDs. Keep `--seed`, `--num-queries`, and `--test-queries` consistent with training. Training also rejects augmentation pairs tied to held-out relevant documents or exact held-out query text.

### 3. Compare RAG thinking modes

A BM25 comparison requires no trained encoder:

```bash
ir-llm-rag --retriever bm25 --num-queries 5 --skip-judge
```

After training both variants, compare BM25 and the distilled encoder on exported test queries:

```bash
ir-llm-rag --retriever both \
  --encoder-model outputs/training/splade_distilled \
  --queries-file outputs/training/test_queries.json \
  --num-queries 10
```

Use `outputs/training/splade` instead if you skipped distillation. The default generator is `Qwen/Qwen3-0.6B`; thinking mode uses its chat template's `enable_thinking` option. Another generator may not support this comparison. Answers and retrieved document IDs are saved under `outputs/rag/`, with judge means and valid-rating counts in `summary.json`. Without `--queries-file`, RAG uses the final dataset topics as a separate exploratory experiment.

The original script entry points remain available:

```bash
python IR_training_evaluation.py --help
python synthetic_data_gen.py --help
python rag_test.py --help
```

## Corpus size and evaluation limits

The default index contains the **first 5,000 corpus documents**, while the default query split selects 900 topics and reserves 100 for evaluation with seed 42. This keeps initial experiments manageable. Some relevant documents fall outside that corpus, which lowers recall and may leave queries without retrievable positives. Scores from this subset are **not official LoTTE benchmark scores**.

To increase the corpus, give all commands the same document limit and a fresh index path:

```bash
ir-llm-train --max-docs 20000 --index-path data/index_lotte_20000
```

Index settings are recorded and checked before reuse. Existing nonempty directories are never automatically deleted. Reusing an older index without a manifest requires you to verify its dataset and corpus size yourself.

SPLADE evaluation scans CPU sparse tensors in batches. It does not use a production inverted index. Sparsity and memory consumption depend on the model, and a larger corpus can still require substantial RAM. Model inputs are truncated to 256 tokens; stored text metadata is limited to 8,192 characters. RAG context is limited to 1,000 characters per retrieved document. Qwen thinking may exceed the generation budget; increase `--max-new-tokens` when needed.

## Code layout

```text
ir_llm/
  config.py       Shared experiment defaults
  data.py         Dataset loading and safe index creation/reuse
  runtime.py      Device selection, random seeds, generator loading
  models.py       SPLADE encoder and cross-encoder teacher
  retrieval.py    Metadata access, sparse retrieval, metric adapters
  training.py     Triplet mining and training objectives
  rag.py          Answer generation and judge parsing
IR_training_evaluation.py   Training/evaluation CLI
synthetic_data_gen.py       Synthetic query CLI
rag_test.py                 RAG comparison CLI (not a unit test)
utils.py                    Compatibility imports for older notebooks
tests/                      Offline regression tests
```

## Development

```bash
ruff check .
ruff format --check .
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 pytest -q
```

Tests exercise held-out splits, augmentation filtering, hard negatives, distillation gradients, sparse retrieval, generated-token decoding, and thinking-mode result labels using small tensors and test doubles. They do not download models or datasets. GitHub Actions runs these checks with Python 3.11 and CPU Torch.

Full dataset training and model-generated evaluation require a separate experiment run; passing regression tests does not establish retrieval or answer quality.

## References

- [LoTTE dataset](https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md)
- [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720)
- [PyTerrier indexing and metadata](https://pyterrier.readthedocs.io/en/stable/terrier/indexing.html)
- [Qwen3-0.6B model and thinking controls](https://huggingface.co/Qwen/Qwen3-0.6B)
