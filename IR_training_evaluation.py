import pandas as pd
import pyterrier as pt
import torch
import torch.nn as nn
import torch.nn.functional as F
import ir_measures
import getpass
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple
from ir_measures import RR, Recall
import os
os.environ["IR_DATASETS_HOME"] = f"/tempory/{os.environ['USER']}/LLM_DATA"
import pyterrier as pt
import ir_datasets
from utils get_best_device,get_text_from_index,get_doc_text,doc_exists_in_index,to_ir_measures_qrels, to_ir_measures_run,find_hard_negatives_bm25,create_training_triplets,CrossEncoderTeacher,distillation_loss,train_splade_with_distillation,
retriever_to_results,RAGPipeline,splade_retriever
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForMaskedLM,AutoModel,SPLADEEncoder,contrastive_loss,train_splade




""""" getting LoTTE data and Setting Index """"

device = get_best_device()
dataset = pt.get_dataset("irds:lotte/technology/dev/search")
print("Dataset loaded: lotte/technology/dev/search")
# Configuration
num_docs = 5000  # Limit documents for practical
num_queries = 900  # Limit queries
# Get queries and qrels first to know which documents are relevant
queries_df = dataset.get_topics().head(num_queries)
qrels_df = dataset.get_qrels()

# Filter qrels for our queries first
query_ids = set(queries_df["qid"].tolist())
qrels_df = qrels_df[qrels_df["qid"].isin(query_ids)].copy()

# Get the set of relevant document IDs we need
relevant_doc_ids = set(qrels_df["docno"].tolist())

print("Queries and Qrels loaded:")
print(f"  - Queries: {len(queries_df)}")
print(f"  - Qrels: {len(qrels_df)}")
# Download the PyTerrier index of Lotte (technology subset)
index_path = Path(f"/tempory/{os.environ['USER']}/LLM_DATA/index_lotte").absolute()

# Initialize PyTerrier with the index
index = pt.terrier.TerrierIndex(str(index_path))




""""" Loading synthetic pata if exists """"
synthetic_training_path = Path("training_data_for_splade_qrels_style_split.json")
synthetic_pairs = []

if synthetic_training_path.exists():
    with open(synthetic_training_path, "r", encoding="utf-8") as f:
        synthetic_data = json.load(f)

    # Convert to query-positive pairs (filter to docs in our index)
    for item in synthetic_data:
        # Check if doc exists in index
        if doc_exists_in_index(index, item["doc_id"]):
            synthetic_pairs.append(
                {
                    "query": item["query"],
                    "doc_id": item["doc_id"],
                    "source": item.get("source", "synthetic"),
                }
            )

    print(f"Loaded {len(synthetic_pairs)} synthetic training pairs from Practical 04")
else:
    print(f"No synthetic training data found at {synthetic_training_path}")
    print(
        "Run Practical 04 first to generate synthetic queries, or continue with qrels only."
    )



# Create training data with hard negatives
num_train_queries = 800


# Create triplets from qrels (original data)
training_triplets = create_training_triplets(
    queries_df,
    qrels_df,
    index,
    bm25,
    num_hard_negatives=3,
    max_queries=num_train_queries,
)

print(f"\nTraining triplets from qrels: {len(training_triplets)}")

# Add triplets from synthetic pairs (if available)
num_train_queries = 500
if synthetic_pairs:
    print(f"Adding triplets from {len(synthetic_pairs)} synthetic pairs...")

    for pair in tqdm(
        synthetic_pairs[:num_train_queries], desc="Processing synthetic pairs"
    ):
        query_text = pair["query"]
        positive_id = pair["doc_id"]
        positive_text = get_doc_text(index, positive_id)

        if not positive_text:
            continue

        # Find hard negatives for synthetic queries too
        hard_neg_ids = find_hard_negatives_bm25(
            query_text, {positive_id}, bm25, num_negatives=3
        )
        hard_neg_texts = get_text_from_index(index, hard_neg_ids)
        hard_neg_list = [
            hard_neg_texts[nid] for nid in hard_neg_ids if nid in hard_neg_texts
        ]

        if hard_neg_list:
            training_triplets.append(
                {
                    "query": query_text,
                    "positive": positive_text,
                    "negatives": hard_neg_list,
                }
            )

    print(f"Total training triplets (qrels + synthetic): {len(training_triplets)}")
if training_triplets:
    print("Example triplet:")
    print(f"  Query: {training_triplets[0]['query'][:60]}...")
    print(f"  Positive: {training_triplets[0]['positive'][:60]}...")
    print(f"  Num negatives: {len(training_triplets[0]['negatives'])}")



splade_baseline_model = "distilbert-base-uncased"    # using distill bert encoder
mlm_tokenizer = AutoTokenizer.from_pretrained(splade_baseline_model)
mlm_model = AutoModelForMaskedLM.from_pretrained(splade_baseline_model)
mlm_model = mlm_model.to(device)
mlm_model.eval()

# training splade on top of bert 
splade_encoder = SPLADEEncoder(splade_baseline_model)
splade_encoder = splade_encoder.to(device)

num_epochs = 30
if training_triplets:
    print(f"Training on {len(training_triplets)} triplets with hard negatives...")
    train_splade(
        splade_encoder,
        training_triplets,
        num_epochs=num_epochs,
        batch_size=4,
        use_hard_negatives=True,
    )
else:
    print("No training triplets available. Skipping SPLADE training.")



# Train SPLADE with distillation
num_epochs = 30 # You can increase this for better performance

cross_encoder = CrossEncoderTeacher("cross-encoder/ms-marco-MiniLM-L6-v2")
cross_encoder = cross_encoder.to(device)
splade_encoder_from_distilation = SPLADEEncoder(splade_baseline_model)
splade_encoder_from_distilation = splade_encoder.to(device)
if training_triplets:
    print(f"Training on {len(training_triplets)} triplets with distillation...")
    train_splade_with_distillation(
        splade_encoder,
        cross_encoder,
        training_triplets,
        num_epochs=num_epochs,
        batch_size=4,
        learning_rate=2e-5,
        temperature=1.0, # Temperature for distillation
        contrastive_temperature=0.05, # Temperature for InfoNCE
        alpha=0.5, # Weight for distillation loss
    )
else:
    print("No training triplets available. Skipping distillation training.")




gen_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)

is_quantized = "AWQ" in gen_model_name or "GPTQ" in gen_model_name

if is_quantized:
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        device_map="auto",
    )
    print(f"Generation model loaded: {gen_model_name} (pre-quantized)")
else:
    gen_model = AutoModelForCausalLM.from_pretrained(
        gen_model_name,
        dtype=torch.float16,
    )
    gen_model = gen_model.to(device)
    print(f"Generation model loaded: {gen_model_name} on {device}")

gen_model.eval()


# Create RAG pipeline
# Use document IDs from qrels (relevant documents in the index)
doc_id_list = list(relevant_doc_ids)

rag_splade = RAGPipeline(
    retriever=splade_encoder,
    generator=gen_model,
    generator_tokenizer=gen_tokenizer,
    pt_index=index,
    doc_ids=doc_id_list,
    top_k=3,
)

rag_distillation_spade = RAGPipeline(
    retriever=splade_encoder_from_distillation,
    generator=gen_model,
    generator_tokenizer=gen_tokenizer,
    pt_index=index,
    doc_ids=doc_id_list,
    top_k=3,
)


splade_results = retriever_to_results(splade_retriever, test_queries)
splade_metrics = ir_measures.calc_aggregate(
    metrics, qrels_ir, to_ir_measures_run(splade_results)
)

print("\n=== SPLADE Results ===")
for metric, value in splade_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Evaluate SPLADE (classic)
test_queries = queries_df.tail(100)  # Use last queries as test
splade_results = rag_splade.retrieve(test_queries, top_k=20)
splade_metrics = ir_measures.calc_aggregate(
    metrics, qrels_ir, to_ir_measures_run(splade_results)
)

print("\n=== SPLADE Classic  Results ===")
for metric, value in splade_metrics.items():
    print(f"  {metric}: {value:.4f}")


# Evaluate SPLADE (distillation)
test_queries = queries_df.tail(100)  # Use last queries as test
splade_results_distill = rag_splade_distill.retrieve(test_queries, top_k=20)
splade_metrics_distill = ir_measures.calc_aggregate(
    metrics, qrels_ir, to_ir_measures_srun(splade_results)
)

print("\n=== SPLADE distill Results ===")
for metric, value in splade_metrics_distill.items():
    print(f"  {metric}: {value:.4f}")


# Compare with BM25200:300
bm25_results = bm25.transform(queries_df[["qid", "query"]])
bm25_metrics = ir_measures.calc_aggregate(
    metrics, qrels_ir, to_ir_measures_run(bm25_results)
)

print("\n=== BM25 Results ===")
for metric, value in bm25_metrics.items():
    print(f"  {metric}: {value:.4f}")




