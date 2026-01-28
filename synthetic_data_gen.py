import shutil
import torch
import pyterrier as pt
import pandas as pd
from pathlib import Path
from collections import Counter
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from rouge_score import rouge_scorer
from evaluate import load as load_metric
from utils get_best_device


device = get_best_device()
# Load a model for generation
#
# Options (uncomment ONE model_name):
#
# 1. SmolLM2-1.7B float16 (~3.4GB) - default, works on all platforms
model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
#
# 2. Pre-quantized models (GPTQ/AWQ) - Linux only, requires:
#    uv pip install auto-gptq autoawq  (or: uv sync --extra quantized)
# model_name = "Qwen/Qwen2.5-3B-Instruct-AWQ"  # 3B AWQ, ~2GB
# model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"  # 7B AWQ, ~4GB

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Detect if model is pre-quantized (AWQ/GPTQ) by name
is_quantized = "AWQ" in model_name or "GPTQ" in model_name

if is_quantized:
    # Pre-quantized models need autoawq/auto-gptq (Linux only)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    print(f"Model loaded: {model_name} (pre-quantized)")
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
    )
    model = model.to(device)
    print(f"Model loaded: {model_name} on {device}")

model.eval()

def build_prompt(user: str, system: str = "You are a helpful assistant.") -> str:
    """Build a chat-format prompt using the tokenizer's chat template."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

@torch.no_grad()
def generate_text(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    do_sample: bool = True,
    num_return_sequences: int = 1,
) -> list[str]:
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode and extract only the response
    generated = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True)
        # Extract the part after the last "assistant"
        if "assistant" in text.lower():
            text = text.split("assistant")[-1].strip()
        generated.append(text)

    return generated

dataset = pt.get_dataset("irds:lotte/technology/dev/search")
irds = dataset.irds_ref()
pt.java.init()
pt.terrier.set_property("querying.parser", "MatchOpQLParser")
queries_df = dataset.get_topics()
qrels_df = dataset.get_qrels()

# Index the entire corpus with PyTerrier (or load existing index)
# We store document text in metadata for retrieval during RAG
index_path = Path(f"/tempory/{os.environ['USER']}/LLM_DATA/index_lotte").absolute()

# Check if index already exists
if (index_path / "data.properties").exists():
    print(f"Loading existing index from {index_path}")
    index_ref = str(index_path)
else:
    print(f"Creating new index at {index_path}")
    if index_path.is_dir():
        shutil.rmtree(index_path)
    index_path.mkdir(parents=True, exist_ok=True)

    indexer = pt.IterDictIndexer(
        str(index_path),
        overwrite=True,
        meta={"docno": 50, "text": 4096},  # Store text in metadata
        meta_reverse=["docno"],
    )

    # Index the corpus - PyTerrier handles iteration efficiently
    print("Indexing corpus...")
    index_ref = indexer.index(dataset.get_corpus_iter())

# Get index statistics
index = pt.IndexFactory.of(index_ref, memory={"meta": True})
meta_index = index.getMetaIndex()

# Helper functions to get document text from index (uses cached meta_index)
def get_doc_text(meta_index, doc_id: str) -> str:
    """Retrieve document text from PyTerrier index metadata."""
    try:
        docid = meta_index.getDocument("docno", doc_id)
        if docid >= 0:
            return meta_index.getItem("text", docid)
    except Exception:
        pass
    return ""


def get_text_from_index(meta_index, doc_ids: list[str]) -> dict[str, str]:
    """Retrieve text for multiple documents from the index metadata."""
    result = {}
    for doc_id in doc_ids:
        try:
            docid = meta_index.getDocument("docno", doc_id)
            if docid >= 0:
                result[doc_id] = meta_index.getItem("text", docid)
        except Exception:
            pass
    return result
    
# Configuration for query generation
num_docs_to_augment = 250 # Number of documents to generate queries for
queries_per_doc = 2  # Number of queries to generate per document

def generate_queries_for_document(
    document: str,
    num_queries: int = 2,
    max_doc_length: int = 500,
) -> List[str]:
    """
    Generate search queries that the document would answer.

    Args:
        document: The document text
        num_queries: Number of queries to generate
        max_doc_length: Maximum document length to use

    Returns:
        List of generated queries
    """
    # Implement query generation

    doc_excerpt = document[:max_doc_length]
    prompt = f"Given this document : {doc_excerpt} , generate {num_queries} search queries that the document would answer , the queries should not be the same  "
    prompt_new = build_prompt(prompt)
    response = generate_text(prompt_new, max_new_tokens=200, temperature=0.1)
    return response

def generate_training_pairs(
    index_ref,
    qrels_df: pd.DataFrame,
    num_docs: int = 1,
    queries_per_doc: int = 2,
) -> List[Dict]:
    """
    Generate query-document training pairs.

    Args:
        index_ref: PyTerrier index reference
        qrels_df: DataFrame with qrels (to get document IDs)
        num_docs: Number of documents to process
        queries_per_doc: Queries to generate per document

    Returns:
        List of {"query": str, "doc_id": str, "document": str}
    """
    # Generate training pairs

    # 1. Sample documents from the corpus (use qrels to get doc IDs)
    # 2. Generate queries for each document
    # 3. Create training pairs
    import random
    doc_ids = qrels_df["docno"].unique().tolist()
    sampled_ids = random.sample(doc_ids, min(num_docs, len(doc_ids)))
    training_pairs = []

    # Use the existing meta_index from the global scope
    global meta_index

    for doc_id in tqdm(sampled_ids):
        doc_text = get_doc_text(meta_index, doc_id)
        if not doc_text:
            continue

        generated_queries = generate_queries_for_document(doc_text, num_queries=queries_per_doc)

        for query in generated_queries:
            training_pairs.append(
                {
                    "query": query,
                    "doc_id": doc_id,
                    "document": doc_text,
                    "source": "synthetic",
                }
            )
    return training_pairs


# Generate synthetic training pairs
synthetic_pairs = generate_training_pairs(
    index_ref,
    qrels_df,
    num_docs=1000,
    queries_per_doc=3,
)
def create_combined_training_data(
    index_ref,
    qrels_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    synthetic_pairs: List[Dict],
) -> List[Dict]:
    """
    Combine real qrels with synthetic pairs.

    Args:
        index_ref: PyTerrier index reference
        qrels_df: Original relevance judgments
        queries_df: Original queries
        synthetic_pairs: Synthetically generated pairs

    Returns:
        Combined list of training pairs
    """
    training_data = []

    # Add real pairs from qrels
    for _, row in qrels_df.iterrows():
        qid = row["qid"]
        doc_id = row["docno"]

        query_rows = queries_df[queries_df["qid"] == qid]["query"].values
        if len(query_rows) == 0:
            continue

        doc_text = get_doc_text(meta_index, doc_id)
        if not doc_text:
            continue

        training_data.append(
            {
                "query": query_rows[0],
                "doc_id": doc_id,
                "document": doc_text,
                "source": "qrels",  # From original dataset
            }
        )

    # Add synthetic pairs
    training_data.extend(synthetic_pairs)

    return training_data

import json

# Prepare data for saving (don't include full document text to save space)
training_export = []
for pair in synthetic_pairs:
    training_export.append(
        {
            "query": pair["query"],
            "doc_id": pair["doc_id"],
            "source": pair["source"],
        }
    )

output_dir = Path("./GENERATED_DATA")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "training_data_for_splade.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(training_export, f, indent=2)

print(f"Saved {len(training_export)} training pairs to {output_path}")



import json
import re

IN_PATH = "GENERATED_DATA/training_data_for_splade.json"
OUT_PATH = "training_data_for_splade_qrels_style_split.json"

def split_numbered_questions(q: str) -> list[str]:
    """
    Split strings like:
      '1. question one\n2. question two'
    into ['question one', 'question two'].

    If it doesn't look like a numbered list, return [q].
    """
    s = q.strip()

    # Detect numbered lines: start of line has digits + dot
    if not re.search(r"(?m)^\s*\d+\.\s+", s):
        return [s]

    # Split on the numbered markers, keep only the content parts
    parts = re.split(r"(?m)^\s*\d+\.\s+", s)
    parts = [p.strip() for p in parts if p.strip()]

    # Clean surrounding quotes if present
    cleaned = []
    for p in parts:
        p = p.strip().strip('"').strip("'").strip()
        if p:
            cleaned.append(p)

    return cleaned if cleaned else [s]


with open(IN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

out = []
for item in data:
    q = item["query"]
    doc_id = item["doc_id"]
    source = item["source"]

    # Only split synthetic combined queries; keep qrels as-is
    if source == "synthetic":
        subqueries = split_numbered_questions(q)
        for sq in subqueries:
            out.append({"query": sq, "doc_id": doc_id, "source": source})
    else:
        out.append(item)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print(f"Saved: {OUT_PATH} | before={len(data)} after={len(out)}")

