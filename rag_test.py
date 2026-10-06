"""Compare RAG answers with Qwen thinking enabled and disabled.

Despite its historical filename this is an experiment command, not a unit test.
"""

import argparse
import json
from pathlib import Path

from ir_llm.config import DEFAULT_DATASET, DEFAULT_GENERATOR, DEFAULT_INDEX, DEFAULT_MAX_DOCS


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--index-path", type=Path, default=DEFAULT_INDEX)
    parser.add_argument(
        "--max-docs", type=int, default=DEFAULT_MAX_DOCS, help="Corpus limit (default: 5000)"
    )
    parser.add_argument("--generator-model", default=DEFAULT_GENERATOR)
    parser.add_argument("--encoder-model", type=Path, help="Saved SPLADE checkpoint directory")
    parser.add_argument("--retriever", choices=("bm25", "splade", "both"), default="bm25")
    parser.add_argument("--queries-file", type=Path, help="JSON query records exported by training")
    parser.add_argument("--num-queries", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-judge", action="store_true", help="Export answers without self-judging"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/rag"))
    return parser


def evaluate_pipeline(pipeline, queries, judge=None, max_new_tokens=1024):
    """Keep each answer and judge result attached to its query and mode."""
    records = []
    for thinking in (False, True):
        pipeline.thinking = thinking
        for query in queries:
            retrieved = pipeline.retrieve(query["query"])
            texts = pipeline._get_texts([doc_id for doc_id, _ in retrieved])
            context = "\n\n".join(texts.get(doc_id, "")[:1000] for doc_id, _ in retrieved)
            answer = pipeline.generate(query["query"], context, max_new_tokens=max_new_tokens)
            record = {
                "qid": str(query["qid"]),
                "query": query["query"],
                "thinking": thinking,
                "retrieved_docs": retrieved,
                "generated_answer": answer,
            }
            if judge is not None:
                record["judge"] = judge(query["query"], answer)
            records.append(record)
    return records


def summarize_scores(records):
    """Report valid judge counts alongside means; failed judgments remain visible."""
    summary = {}
    for thinking in (False, True):
        rows = [row for row in records if row["thinking"] == thinking]
        scores = {}
        for metric in ("relevance", "helpfulness"):
            values = [
                row.get("judge", {}).get(metric)
                for row in rows
                if type(row.get("judge", {}).get(metric)) is int and 1 <= row["judge"][metric] <= 5
            ]
            scores[metric] = {
                "mean": sum(values) / len(values) if values else None,
                "valid": len(values),
                "total": len(rows),
            }
        summary["thinking" if thinking else "no_thinking"] = scores
    return summary


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("num_queries", "top_k", "max_new_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_docs is not None and args.max_docs < 1:
        parser.error("--max-docs must be positive")
    if args.retriever in ("splade", "both") and (
        args.encoder_model is None or not args.encoder_model.is_dir()
    ):
        parser.error("--encoder-model must point to a saved SPLADE checkpoint")

    from ir_llm.data import index_document_ids, load_dataset, load_or_create_index
    from ir_llm.models import SPLADEEncoder
    from ir_llm.rag import RAGPipelineBM25, RAGPipelineSplade, judge_answer_quality
    from ir_llm.runtime import load_generator, select_device, set_seed

    set_seed(args.seed)
    device = select_device(args.device)
    dataset, topics, _ = load_dataset(args.dataset)
    index = load_or_create_index(dataset, args.index_path, max_docs=args.max_docs)
    if args.queries_file:
        queries = json.loads(args.queries_file.read_text(encoding="utf-8"))[: args.num_queries]
    else:
        queries = topics.tail(args.num_queries)[["qid", "query"]].to_dict("records")
    if not queries:
        parser.error("No queries available for evaluation")
    tokenizer, generator = load_generator(args.generator_model, device)
    common = dict(
        generator=generator,
        generator_tokenizer=tokenizer,
        pt_index=index,
        doc_ids=[],
        top_k=args.top_k,
    )
    pipelines = {}
    if args.retriever in ("bm25", "both"):
        pipelines["bm25"] = RAGPipelineBM25(retriever=index.retriever("BM25"), **common)
    if args.retriever in ("splade", "both"):
        config_path = args.encoder_model / "splade_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        encoder = SPLADEEncoder(
            str(args.encoder_model), sparsity_weight=config.get("sparsity_weight", 0.0001)
        ).to(device)
        encoder.eval()
        common["doc_ids"] = index_document_ids(index)
        pipelines["splade"] = RAGPipelineSplade(retriever=encoder, device=device, **common)

    judge = None
    if not args.skip_judge:

        def judge(query, answer):
            return judge_answer_quality(generator, tokenizer, query, answer)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, pipeline in pipelines.items():
        records = evaluate_pipeline(pipeline, queries, judge, args.max_new_tokens)
        (args.output_dir / f"{name}_answers.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary[name] = summarize_scores(records)
    report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "scores": summary,
        "evaluation": "Same-model judgments are exploratory ratings, not ground-truth accuracy.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
