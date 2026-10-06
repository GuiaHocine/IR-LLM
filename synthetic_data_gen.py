"""Generate synthetic search-query/document pairs for SPLADE training.

Generation uses documents relevant to the training query split. Documents relevant
to held-out queries are excluded. Importing this module has no model side effects.
"""

import argparse
import json
import random
import re
from pathlib import Path

from ir_llm.config import DEFAULT_DATASET, DEFAULT_INDEX, DEFAULT_MAX_DOCS
from IR_training_evaluation import positive_int, split_queries


def split_numbered_questions(text):
    """Parse one-question-per-line output, including numbered or bulleted lists."""
    text = text.strip()
    if not text:
        return []
    lines = text.splitlines()
    marker = r"^\s*(?:\d+[.)]\s*|[-*]\s+)"
    has_list = any(re.match(marker, line) for line in lines)
    questions = []
    for line in lines:
        if has_list and not re.match(marker, line):
            continue
        line = re.sub(marker, "", line).strip()
        line = line.strip("\"'").strip()
        if line and line.casefold() not in {question.casefold() for question in questions}:
            questions.append(line)
    return questions


def build_prompt(tokenizer, document, num_queries=2, max_doc_length=1500):
    messages = [
        {
            "role": "system",
            "content": "You write search queries grounded in the supplied document.",
        },
        {
            "role": "user",
            "content": (
                f"Write exactly {num_queries} distinct search questions that this document answers. "
                "Return only the questions, one numbered question per line, with no introduction.\n\n"
                f"Document:\n{document[:max_doc_length]}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def generate_text(model, tokenizer, prompt, max_new_tokens=200):
    """Decode only the newly generated tokens, preserving the actual answer text."""
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        outputs[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    ).strip()


def generate_queries_for_document(model, tokenizer, document, num_queries=2, max_doc_length=1500):
    prompt = build_prompt(tokenizer, document, num_queries, max_doc_length)
    response = generate_text(model, tokenizer, prompt, max_new_tokens=max(100, num_queries * 60))
    return split_numbered_questions(response)[:num_queries]


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
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-1.7B-Instruct")
    parser.add_argument("--num-docs", type=positive_int, default=100)
    parser.add_argument("--queries-per-doc", type=positive_int, default=2)
    parser.add_argument("--num-queries", type=positive_int, default=900)
    parser.add_argument("--test-queries", type=positive_int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/synthetic_pairs.json"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def select_training_documents(train_qrels, test_qrels, num_docs, seed=42, available_doc_ids=None):
    """Avoid generating augmentation from any held-out relevant document."""
    excluded = set(test_qrels.loc[test_qrels["label"] > 0, "docno"].astype(str))
    candidates = sorted(
        set(train_qrels.loc[train_qrels["label"] > 0, "docno"].astype(str)) - excluded
    )
    if available_doc_ids is not None:
        available = {str(doc_id) for doc_id in available_doc_ids}
        candidates = [doc_id for doc_id in candidates if doc_id in available]
    return random.Random(seed).sample(candidates, min(num_docs, len(candidates)))


def run(args):
    import torch
    from tqdm.auto import tqdm

    from ir_llm.data import index_document_ids, load_dataset, load_or_create_index
    from ir_llm.retrieval import get_doc_text
    from ir_llm.runtime import load_generator, select_device

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dataset, topics, qrels = load_dataset(args.dataset)
    train_topics, test_topics = split_queries(
        topics.head(args.num_queries), args.test_queries, args.seed
    )
    train_qrels = qrels[qrels["qid"].isin(train_topics["qid"])]
    test_qrels = qrels[qrels["qid"].isin(test_topics["qid"])]
    index = load_or_create_index(dataset, args.index_path, max_docs=args.max_docs)
    doc_ids = select_training_documents(
        train_qrels, test_qrels, args.num_docs, args.seed, index_document_ids(index)
    )
    if not doc_ids:
        raise ValueError("No eligible training documents remain after excluding held-out documents")
    # Filter before loading a potentially large generator.
    documents = [(doc_id, get_doc_text(index, doc_id)) for doc_id in doc_ids]
    documents = [(doc_id, text) for doc_id, text in documents if text]
    if not documents:
        raise ValueError("None of the selected documents occur in the index; increase --max-docs")
    tokenizer, model = load_generator(args.model, select_device(args.device))
    pairs = []
    for doc_id, document in tqdm(documents, desc="Generating queries"):
        for query in generate_queries_for_document(
            model, tokenizer, document, args.queries_per_doc
        ):
            pairs.append({"query": query, "doc_id": doc_id, "source": "synthetic"})
    if not pairs:
        raise ValueError("The model produced no queries")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pairs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    manifest.update(
        {
            "training_query_ids": train_topics["qid"].astype(str).tolist(),
            "test_query_ids": test_topics["qid"].astype(str).tolist(),
            "num_pairs": len(pairs),
        }
    )
    args.output.with_suffix(".config.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(pairs)} training pairs to {args.output}")
    return pairs


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.test_queries >= args.num_queries:
        parser.error("--test-queries must be smaller than --num-queries")
    try:
        run(args)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
