import pandas as pd
import os
os.environ["HF_HOME"] = f"/tempory/{os.environ['USER']}/HF_CACHE"
os.environ["TRANSFORMERS_CACHE"] = f"/tempory/{os.environ['USER']}/hf_cache_clean"
os.environ["HF_HUB_CACHE"] = f"/tempory/{os.environ['USER']}/hf_cache_clean"
os.environ["IR_DATASETS_HOME"] = f"/tempory/{os.environ['USER']}/LLM_DATA"
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
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForMaskedLM,AutoModel
from utils import SPLADEEncoder,get_text_from_index, get_doc_text,doc_exists_in_index,to_ir_measures_qrels,to_ir_measures_run,clean_query,RAGPipelineSplade,RAGPipelineBM25,judge_answer_quality,strip_think
import re
from tqdm import tqdm
import pyterrier as pt
import ir_datasets




device = get_best_device()



""""loading the decoder model""""
thinking_model_name = "Qwen/Qwen3-0.6B"
model_name = "Qwen/Qwen3-0.6B"
# load the tokenizer and the model
thinking_tokenizer = AutoTokenizer.from_pretrained(model_name)
thinking_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

thinking_model.eval()
device = next(thinking_model.parameters()).device


"""" loading the LoTTE Data and index """""
dataset = pt.get_dataset("irds:lotte/technology/dev/search")
num_docs = 5000 
num_queries = 900
queries_df = dataset.get_topics().head(num_queries)
qrels_df = dataset.get_qrels()
query_ids = set(queries_df["qid"].tolist())
qrels_df = qrels_df[qrels_df["qid"].isin(query_ids)].copy()
relevant_doc_ids = set(qrels_df["docno"].tolist())
index_path = Path(f"/tempory/{os.environ['USER']}/LLM_DATA/index_lotte").absolute()
index = pt.terrier.TerrierIndex(str(index_path))

""" Loading the splade IR model that has been trained in practical 5 """"
# 1. Re-instantiate the class
hyperparams = torch.load("splade_model/hyperparams.pt")
splade_encoder = SPLADEEncoder(sparsity_weight=hyperparams["sparsity_weight"])

# 2. Load the weights
splade_encoder.load_state_dict(torch.load("splade_model/model_weights.pt"))
splade_encoder.eval() # Set to evaluation mode


"""" RAG USING BM25 RETRIEVAL WITH & WITHOUT THINKING (COT) AND EVALUATED USING SELF CONSISTENCY (LLM AS JUDGE) """"


bm25 = index.retriever("BM25")
doc_id_list = list(relevant_doc_ids)
rag_bm25_no_thinking = RAGPipelineBM25(
    retriever=bm25,
    generator=thinking_model,      
    generator_tokenizer=thinking_tokenizer,
    pt_index=index,
    doc_ids=doc_id_list, 
    top_k=3,
    thinking=False,
)
rag_bm25_thinking = RAGPipelineBM25(
    retriever=bm25,
    generator=thinking_model,      
    generator_tokenizer=thinking_tokenizer,
    pt_index=index,
    doc_ids=doc_id_list, 
    top_k=3,
    thinking=True,
)

results_thinking = []
results_no_thinking=[]

last_queries = queries_df.tail(50)

for _, row in tqdm(
    last_queries.iterrows(),
    total=len(last_queries),
    desc="Generating RAG answers"
):
    query = row["query"]

    out_NT = rag_bm25_no_thinking(query)["generated_answer"]
    out_T = rag_bm25_thinking(query)["generated_answer"]

    # Strip thinking
    if "</think>" in out_T:
        out_T = out_T.split("</think>")[-1].strip()

    results_thinking.append({
        "qid": row["qid"],
        "query": query,
        "generated_answer": out_NT
    })
    results_no_thinking.append({
        "qid": row["qid"],
        "query": query,
        "generated_answer": out_T
    })
rag_results_bm25_thinking = pd.DataFrame(results_thinking)
rag_results_bm25_no_thinking = pd.DataFrame(result_no_thinking)

judge_scores_thinking = []
judge_scores_no_thinking = []

for _, row in tqdm(rag_results_bm25_thinking.iterrows(), total=len(rag_results_bm25_thinking), desc="LLM-as-Judge"):
    judge_scores_thinking = judge_answer_quality(
        judge_model=thinking_model,              # or your judge model
        judge_tokenizer=thinking_tokenizer,      # or judge tokenizer
        query=row["query"],
        answer=row["generated_answer"],
        max_new_tokens=256,
    )
    judge_scores_thinking.append(judge_scores_thinking)

for _, row in tqdm(rag_results_bm25_no_thinking.iterrows(), total=len(rag_results_bm25_no_thinking), desc="LLM-as-Judge"):
    judge_scores_no_thinking = judge_answer_quality(
        judge_model=thinking_model,              # or your judge model
        judge_tokenizer=thinking_tokenizer,      # or judge tokenizer
        query=row["query"],
        answer=row["generated_answer"],
        max_new_tokens=256,
    )
    judge_scores_no_thinking.append(judge_scores_no_thinking)


judge_df_thinking = pd.DataFrame(judge_scores_thinking)
scored_df_thinking = pd.concat([rag_results_bm25_thinking .reset_index(drop=True), judge_df_thinking], axis=1)
final_result_thinking_bm25=scored_df_thinking[["relevance", "helpfulness"]].mean()

judge_df_no_thinking = pd.DataFrame(judge_scores_no_thinking)
scored_df__no_thinking = pd.concat([rag_results_bm25_no_thinking .reset_index(drop=True), judge_df_no_thinking], axis=1)
final_result_no_thinking_bm25=scored_df_thinking[["relevance", "helpfulness"]].mean()




"""" RAG USING SPLADE TRAINED RETRIEVAL by distilling cross-encoder  WITH & WITHOUT THINKING (COT) AND EVALUATED USING SELF CONSISTENCY (LLM AS JUDGE) """"

doc_id_list = list(relevant_doc_ids)

rag_splade_no_thinking = RAGPipelineSplade(
    retriever=splade_encoder,
    generator=thinking_model,          
    generator_tokenizer=thinking_tokenizer,
    pt_index=index,
    doc_ids=doc_id_list,              
    top_k=3,
    thinking=False,
    device = device
)

rag_splade_thinking = RAGPipelineSplade(
    retriever=splade_encoder,
    generator=thinking_model,          
    generator_tokenizer=thinking_tokenizer,
    pt_index=index,
    doc_ids=doc_id_list,             
    top_k=3,
    thinking=True,
    device = device
)


results_thinking = []
results_no_thinking =[]

last_queries = queries_df.tail(50)

for _, row in tqdm(
    last_queries.iterrows(),
    total=len(last_queries),
    desc="Generating RAG answers"
):
    query = row["query"]

    out_NT = rag_splade_no_thinking(query)["generated_answer"]
    out_T = rag_splade_thinking(query)["generated_answer"]

    # Strip thinking
    if "</think>" in out_T:
        out_T = out_T.split("</think>")[-1].strip()

    results_thinking.append({
        "qid": row["qid"],
        "query": query,
        "generated_answer": out_NT
    })
    results_no_thinking.append({
        "qid": row["qid"],
        "query": query,
        "generated_answer": out_T
    })
rag_results_splade_thinking = pd.DataFrame(results_thinking)
rag_results_splade_no_thinking = pd.DataFrame(result_no_thinking)

judge_scores_thinking = []
judge_scores_no_thinking = []

for _, row in tqdm(rag_results_bm25_thinking.iterrows(), total=len(rag_results_splade_thinking), desc="LLM-as-Judge"):
    judge_scores_thinking = judge_answer_quality(
        judge_model=thinking_model,              # or your judge model
        judge_tokenizer=thinking_tokenizer,      # or judge tokenizer
        query=row["query"],
        answer=row["generated_answer"],
        max_new_tokens=256,
    )
    judge_scores_thinking.append(judge_scores_thinking)

for _, row in tqdm(rag_results_bm25_no_thinking.iterrows(), total=len(rag_results_splade_no_thinking), desc="LLM-as-Judge"):
    judge_scores_no_thinking = judge_answer_quality(
        judge_model=thinking_model,              # or your judge model
        judge_tokenizer=thinking_tokenizer,      # or judge tokenizer
        query=row["query"],
        answer=row["generated_answer"],
        max_new_tokens=256,
    )
    judge_scores_no_thinking.append(judge_scores_no_thinking)


judge_df_thinking = pd.DataFrame(judge_scores_thinking)
scored_df_thinking = pd.concat([rag_results_bm25_thinking .reset_index(drop=True), judge_df_thinking], axis=1)
final_result_thinking_splade=scored_df_thinking[["relevance", "helpfulness"]].mean()

judge_df_no_thinking = pd.DataFrame(judge_scores_no_thinking)
scored_df__no_thinking = pd.concat([rag_results_bm25_no_thinking .reset_index(drop=True), judge_df_no_thinking], axis=1)
final_result_no_thinking_splade=scored_df_thinking[["relevance", "helpfulness"]].mean()








