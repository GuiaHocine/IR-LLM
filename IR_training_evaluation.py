"""Train SPLADE variants and evaluate them against a BM25 baseline.

Run ``python IR_training_evaluation.py --help`` for configuration options.
Importing this module does not load models, datasets, or start training.
"""

import argparse
import json
import random
from pathlib import Path

from ir_llm.config import DEFAULT_DATASET, DEFAULT_INDEX, DEFAULT_MAX_DOCS


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--max-docs",
        type=positive_int,
        default=DEFAULT_MAX_DOCS,
        help="Corpus limit (default: 5000)",
    )
    parser.add_argument("--num-queries", type=positive_int, default=900)
    parser.add_argument("--test-queries", type=positive_int, default=100)
    parser.add_argument("--encoder-model", default="distilbert-base-uncased")
    parser.add_argument("--teacher-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--hard-negatives", type=positive_int, default=3)
    parser.add_argument("--synthetic-data", type=Path, help="Optional JSON query/doc_id pairs")
    parser.add_argument("--skip-distillation", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def split_queries(topics, test_count, seed=42):
    """Split unique query IDs before mining training data or evaluating runs."""
    if topics["qid"].duplicated().any():
        raise ValueError("Topics must contain unique query IDs")
    if test_count <= 0 or test_count >= len(topics):
        raise ValueError("test-queries must be smaller than the number of available queries")
    shuffled = topics.sample(frac=1, random_state=seed).reset_index(drop=True)
    return shuffled.iloc[test_count:].copy(), shuffled.iloc[:test_count].copy()


def load_synthetic_pairs(path, excluded_doc_ids=(), excluded_queries=()):
    """Validate pairs and remove documents/queries reserved for evaluation."""
    with Path(path).open(encoding="utf-8") as stream:
        pairs = json.load(stream)
    if not isinstance(pairs, list):
        raise ValueError("Synthetic data must be a JSON list of query/doc_id objects")
    excluded_doc_ids = {str(doc_id) for doc_id in excluded_doc_ids}
    excluded_queries = {query.strip().casefold() for query in excluded_queries}
    result = []
    seen = set()
    for pair in pairs:
        if not isinstance(pair, dict) or not isinstance(pair.get("query"), str):
            raise ValueError("Every synthetic pair must have a string query and doc_id")
        if "doc_id" not in pair or not isinstance(pair["doc_id"], (str, int)):
            raise ValueError("Every synthetic pair must have a string or integer doc_id")
        query, doc_id = pair["query"].strip(), str(pair["doc_id"])
        key = (query.casefold(), doc_id)
        if (
            query
            and doc_id not in excluded_doc_ids
            and key[0] not in excluded_queries
            and key not in seen
        ):
            result.append({"query": query, "doc_id": doc_id})
            seen.add(key)
    return result


def save_encoder(encoder, path):
    """Save a reusable Hugging Face model, tokenizer, and SPLADE settings."""
    path.mkdir(parents=True, exist_ok=True)
    encoder.model.save_pretrained(path)
    encoder.tokenizer.save_pretrained(path)
    (path / "splade_config.json").write_text(
        json.dumps({"sparsity_weight": encoder.sparsity_weight}, indent=2) + "\n",
        encoding="utf-8",
    )


def run(args):
    import ir_measures
    import torch
    from ir_measures import RR, Recall

    from ir_llm.data import index_document_ids, load_dataset, load_or_create_index
    from ir_llm.models import CrossEncoderTeacher, SPLADEEncoder
    from ir_llm.retrieval import (
        SPLADERetriever,
        find_hard_negatives_bm25,
        get_doc_text,
        get_text_from_index,
        retriever_to_results,
        to_ir_measures_qrels,
        to_ir_measures_run,
    )
    from ir_llm.runtime import select_device
    from ir_llm.training import (
        create_training_triplets,
        train_splade,
        train_splade_with_distillation,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    dataset, topics, qrels = load_dataset(args.dataset)
    topics = topics.head(args.num_queries).copy()
    train_topics, test_topics = split_queries(topics, args.test_queries, args.seed)
    train_qrels = qrels[qrels["qid"].isin(train_topics["qid"])].copy()
    test_qrels = qrels[qrels["qid"].isin(test_topics["qid"])].copy()
    index = load_or_create_index(dataset, args.index_path, max_docs=args.max_docs)
    doc_ids = index_document_ids(index)
    if not doc_ids:
        raise ValueError("The index contains no documents")
    bm25 = index.retriever("BM25", num_results=100)
    triplets = create_training_triplets(
        train_topics,
        train_qrels,
        index,
        bm25,
        num_hard_negatives=args.hard_negatives,
        max_queries=None,
    )
    if args.synthetic_data:
        held_out_docs = test_qrels.loc[test_qrels["label"] > 0, "docno"]
        synthetic_pairs = load_synthetic_pairs(
            args.synthetic_data, held_out_docs, test_topics["query"]
        )
        for pair in synthetic_pairs:
            positive = get_doc_text(index, pair["doc_id"])
            if not positive:
                continue
            negatives = find_hard_negatives_bm25(
                pair["query"], {pair["doc_id"]}, bm25, args.hard_negatives
            )
            texts = get_text_from_index(index, negatives)
            negative_texts = [texts[doc_id] for doc_id in negatives if texts.get(doc_id)]
            if negative_texts:
                triplets.append(
                    {"query": pair["query"], "positive": positive, "negatives": negative_texts}
                )
    if len(triplets) < 2:
        raise ValueError(
            "At least two training triplets are required; increase the corpus/query limits"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_topics.to_csv(args.output_dir / "train_queries.tsv", sep="\t", index=False)
    test_topics.to_csv(args.output_dir / "test_queries.tsv", sep="\t", index=False)
    for name, frame in (("train", train_topics), ("test", test_topics)):
        (args.output_dir / f"{name}_queries.json").write_text(
            frame[["qid", "query"]].to_json(orient="records", indent=2) + "\n",
            encoding="utf-8",
        )
    configuration = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Training queries: {len(train_topics)}; test queries: {len(test_topics)}; triplets: {len(triplets)}"
    )
    measures = [RR @ 10, Recall @ 20]
    qrels_ir = to_ir_measures_qrels(test_qrels)
    metrics = {}
    training_history = {}

    def evaluate(name, results):
        results.to_csv(args.output_dir / f"{name}_run.tsv", sep="\t", index=False)
        aggregate = ir_measures.calc_aggregate(measures, qrels_ir, to_ir_measures_run(results))
        metrics[name] = {str(metric): value for metric, value in aggregate.items()}
        print(f"{name}: {metrics[name]}")

    evaluate("bm25", bm25.transform(test_topics[["qid", "query"]]))
    # Each variant starts from the same pretrained backbone, with its own weights.
    for name in ["splade"] + ([] if args.skip_distillation else ["splade_distilled"]):
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        encoder = SPLADEEncoder(args.encoder_model).to(device)
        if name == "splade":
            training_history[name] = train_splade(
                encoder,
                triplets,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                use_hard_negatives=True,
            )
        else:
            teacher = CrossEncoderTeacher(args.teacher_model).to(device)
            training_history[name] = train_splade_with_distillation(
                encoder,
                teacher,
                triplets,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
            )
            del teacher
        encoder.eval()
        save_encoder(encoder, args.output_dir / name)
        retriever = SPLADERetriever(encoder, index, doc_ids, top_k=20)
        evaluate(name, retriever_to_results(retriever.retrieve, test_topics))
        del retriever, encoder
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training_history.json").write_text(
        json.dumps(training_history, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for contrastive training")
    if args.test_queries >= args.num_queries:
        parser.error("--test-queries must be smaller than --num-queries")
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
