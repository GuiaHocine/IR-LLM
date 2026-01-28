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
import json
import pyterrier as pt
from transformers import AutoTokenizer, AutoModelForCausalLM,AutoModelForMaskedLM,AutoModel
from tqdm import tqdm
import re 
from torch.optim import AdamW

class SPLADEEncoder(nn.Module):
    """
    SPLADE encoder based on an MLM model.

    Architecture:
    - Backbone: Pre-trained MLM model (DistilBERT, etc.)
    - Sparsification: ReLU + log(1 + x)
    - Pooling: Max over positions
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        sparsity_weight: float = 0.0001,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.sparsity_weight = sparsity_weight
        self.vocab_size = self.tokenizer.vocab_size

    def encode(
        self,
        texts: List[str],
        max_length: int = 256,
    ) -> torch.Tensor:
        """
        Encode a list of texts into SPLADE representations.

        Args:
            texts: List of texts
            max_length: Maximum length

        Returns:
            Tensor of shape (batch_size, vocab_size)
        """
        # Implement batch encoding

        # 1. Tokenize input
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(self.model.device)

        # 2. Get MLM logits
        outputs = self.model(**inputs)
        # Shape: (batch_size, sequence_length, vocab_size)
        logits = outputs.logits

        # 3. Apply ReLU then log(1 + x)
        # log(1 + ReLU(logits))
        sparse_rep = torch.log(1 + F.relu(logits))

        # 4. Max-pooling over positions (attention to mask!)
        # We need to mask out padded tokens from max-pooling
        # The attention mask has 1 for real tokens, 0 for padding
        # (batch_size, sequence_length, 1)
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        # Apply mask: set masked positions to a very small negative number so they don't affect max
        masked_sparse_rep = sparse_rep.masked_fill(attention_mask == 0, -torch.inf)

        # Max-pooling over the sequence_length dimension
        # Resulting shape: (batch_size, vocab_size)
        max_pooled_rep = torch.max(masked_sparse_rep, dim=1).values

        return max_pooled_rep


    def compute_flops_loss(self, sparse_rep: torch.Tensor) -> torch.Tensor:
        """
        Compute FLOPS penalty to encourage sparsity.

        FLOPS = sum_j (sum_i w_ij)^2

        This penalty encourages sparse representations.
        """
        # Sum over batch then square
        flops = (sparse_rep.sum(dim=0) ** 2).sum()
        return self.sparsity_weight * flops

    def forward(
        self,
        query_texts: List[str],
        doc_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Returns:
            query_rep: Query representations
            doc_rep: Document representations
            flops_loss: Sparsity penalty
        """
        query_rep = self.encode(query_texts)
        doc_rep = self.encode(doc_texts)

        flops_loss = self.compute_flops_loss(
            query_rep
        ) + self.compute_flops_loss(
            doc_rep
        )

        return query_rep, doc_rep, flops_loss

def addition():
    print("haha")
    
def get_text_from_index(
    index: pt.terrier.TerrierIndex, doc_ids: List[str]
) -> Dict[str, str]:
    """
    Retrieve document text from the index metadata.

    Args:
        index: PyTerrier index reference
        doc_ids: List of document IDs

    Returns:
        Dict mapping doc_id -> text
    """
    meta_index = index.meta_index()

    result = {}
    for doc_id in doc_ids:
        try:
            # Get internal docid from docno
            docid = meta_index.getDocument("docno", doc_id)
            if docid >= 0:
                text = meta_index.getItem("text", docid)
                result[doc_id] = text
        except Exception:
            pass  # Document not found

    return result


# Helper for single document lookup
def get_doc_text(index, doc_id: str) -> str:
    """Get text for a single document from the index."""
    texts = get_text_from_index(index, [doc_id])
    return texts.get(doc_id, "")


def doc_exists_in_index(index, doc_id: str) -> bool:
    """Check if a document exists in the index (without fetching text)."""
    meta_index = index.meta_index()
    try:
        return meta_index.getDocument("docno", doc_id) >= 0
    except Exception:
        return False

def to_ir_measures_qrels(qrels_df):
    """Convert PyTerrier qrels to ir-measures format."""
    return qrels_df.rename(
        columns={"qid": "query_id", "docno": "doc_id", "label": "relevance"}
    )


def to_ir_measures_run(run_df):
    """Convert PyTerrier run to ir-measures format."""
    return run_df.rename(columns={"qid": "query_id", "docno": "doc_id"})

def clean_query(text):
    # Replace hyphens and other special chars with a space
    # This keeps the words but removes the "operator" context
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove extra whitespace
    return " ".join(cleaned.split())


class RAGPipelineSplade:
    """
    Complete RAG pipeline: Retrieval + Generation.

    Uses PyTerrier index for document storage (scalable to large collections).
    """

    def __init__(
        self,
        retriever: SPLADEEncoder,
        generator,
        generator_tokenizer,
        pt_index,
        doc_ids: List[str],
        top_k: int = 3,
        thinking:bool = False,
        device:str = "cpu"
    ):
        super().__init__()
        self.retriever = retriever
        self.generator = generator
        self.gen_tokenizer = generator_tokenizer
        self.pt_index = pt_index
        self.doc_ids = doc_ids
        self.top_k = top_k
        self.thinking = thinking
        self.device = device 
        # Pre-compute document representations
        self._index_documents()

    def _get_doc_text(self, doc_id: str) -> str:
        """Get document text from PyTerrier index."""
        return get_doc_text(self.pt_index, doc_id)

    def _get_texts(self, doc_ids: List[str]) -> Dict[str, str]:
        """Get multiple document texts from PyTerrier index."""
        return get_text_from_index(self.pt_index, doc_ids)

    def _index_documents(self, batch_size: int = 16):
        """Index all documents with SPLADE."""
        print("Indexing documents with SPLADE...")

        # Get texts from PyTerrier index in batches
        all_reps = []

        with torch.no_grad():
            for i in tqdm(range(0, len(self.doc_ids), batch_size)):
                batch_ids = self.doc_ids[i : i + batch_size]
                batch_texts = self._get_texts(batch_ids)
                # Truncate texts and maintain order
                texts = [batch_texts.get(doc_id, "")[:500] for doc_id in batch_ids]
                reps = self.retriever.encode(texts)
                all_reps.append(reps.cpu())

        self.doc_reps = torch.cat(all_reps, dim=0)
        print(f"Documents indexed: {len(self.doc_ids)}")

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant documents.

        Args:
            query: User question
            top_k: Number of documents to retrieve

        Returns:
            List of (doc_id, score)
        """
        if top_k is None:
            top_k = self.top_k

        # Encode query
        with torch.no_grad():
            query_rep = self.retriever.encode([query]).cpu()

        # Compute scores
        scores = torch.matmul(query_rep, self.doc_reps.T).squeeze(0)

        # Top-k
        top_indices = torch.topk(scores, k=min(top_k, len(scores))).indices

        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = scores[idx].item()
            results.append((doc_id, score))

        return results

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 200,
    ) -> str:
        """Generate a response based on context."""
        system_msg = "You are a helpful assistant that answers technical questions using the provided context."
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,enable_thinking=self.thinking
        )
        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.gen_tokenizer.eos_token_id,
            )

        response = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        return response

    def __call__(self, query: str) -> Dict:
        """
        Complete pipeline: retrieval + generation.

        Args:
            query: User question

        Returns:
            Dict with retrieved_docs and generated_answer
        """
        # Implement RAG pipeline

        # 1. Retrieve top-k documents
        retrieved_docs = self.retrieve(query)

        # 2. Build context from documents
        doc_texts = self._get_texts([doc_id for doc_id, _ in retrieved_docs])
        context = "\n\n".join([doc_texts[doc_id][:300] for doc_id, _ in retrieved_docs])

        # 3. Generate response
        generated_answer = self.generate(query, context)

        return {"query": query, "retrieved_docs": retrieved_docs, "generated_answer": generated_answer}


class RAGPipelineBM25:
    """
    Same RAG pipeline as RAGPipelineSplade, but retrieval uses BM25 (PyTerrier).
    Everything else (generation, context building, thinking flag) stays the same.
    """

    def __init__(
        self,
        retriever,  # <-- PyTerrier BM25 retriever (e.g., pt.BatchRetrieve)
        generator,
        generator_tokenizer,
        pt_index,
        doc_ids: List[str],   # kept for API compatibility (not used by BM25)
        top_k: int = 3,
        thinking: bool = False,
    ):
        super().__init__()
        self.retriever = retriever
        self.generator = generator
        self.gen_tokenizer = generator_tokenizer
        self.pt_index = pt_index
        self.doc_ids = doc_ids  # not needed for BM25 retrieval, but kept
        self.top_k = top_k
        self.thinking = thinking

        # BM25 does NOT need doc pre-indexing in torch
        self._index_documents()

    def _get_doc_text(self, doc_id: str) -> str:
        """Get document text from PyTerrier index."""
        return get_doc_text(self.pt_index, doc_id)

    def _get_texts(self, doc_ids: List[str]) -> Dict[str, str]:
        """Get multiple document texts from PyTerrier index."""
        return get_text_from_index(self.pt_index, doc_ids)

    def _index_documents(self, batch_size: int = 16):
        """No-op for BM25 (kept so your code structure doesn't change)."""
        print("BM25 retriever: no document pre-indexing needed.")

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant documents using BM25.

        Returns:
            List of (doc_id, score)
        """
        if top_k is None:
            top_k = self.top_k

        # PyTerrier returns a dataframe with columns: docno, score, rank, qid, ...
        run_df = self.retriever.search(query).head(top_k)

        results = []
        for _, row in run_df.iterrows():
            results.append((row["docno"], float(row["score"])))

        return results

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 1000,
    ) -> str:
        """Generate a response based on context."""
        system_msg = "You are a helpful assistant that answers technical questions using the provided context."
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.gen_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
        )

        # IMPORTANT: use the model's device, not a global "device" unless you defined it
        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to(self.generator.device)

        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.gen_tokenizer.eos_token_id,
            )

        response = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "<think>" in response:
            response = response.split("</think>")[-1].strip()

        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        return response

    def __call__(self, query: str) -> Dict:
        """
        Complete pipeline: retrieval + generation.
        """
        # 1. Retrieve top-k documents
        retrieved_docs = self.retrieve(query)

        # 2. Build context from documents
        doc_texts = self._get_texts([doc_id for doc_id, _ in retrieved_docs])
        context = "\n\n".join([doc_texts.get(doc_id, "")[:300] for doc_id, _ in retrieved_docs])

        # 3. Generate response
        generated_answer = self.generate(query, context)

        return {"query": query, "retrieved_docs": retrieved_docs, "generated_answer": generated_answer}



def strip_think(txt: str) -> str:
    if "</think>" in txt:
        txt = txt.split("</think>")[-1].strip()
    return txt

def judge_answer_quality(
    judge_model,
    judge_tokenizer,
    query: str,
    answer: str,
    max_new_tokens: int = 256,
) -> dict:
    """
    Returns JSON dict:
    {"relevance":1..5, "helpfulness":1..5, "notes":"..."}
    """

    system_msg = (
        "You are an expert evaluator of question-answering systems. "
        "Output ONLY valid JSON."
    )

    user_msg = f"""
Question:
{query}

Answer:
{answer}

Rate from 1 (very poor) to 5 (excellent):
- relevance: does the answer address the question appropriately?
- helpfulness: is it clear, actionable, and complete?

Return ONLY this JSON:
{{
  "relevance": <int>,
  "helpfulness": <int>,
  "notes": "<short justification>"
}}
"""

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    prompt = judge_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # important
    )

    device = next(judge_model.parameters()).device
    inputs = judge_tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.0,   # deterministic judge
            do_sample=False,
            pad_token_id=judge_tokenizer.eos_token_id,
        )

    text = judge_tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = strip_think(text)

    # Robust JSON extraction
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {"relevance": None, "helpfulness": None, "notes": "JSON parse failed", "raw": text}

    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"relevance": None, "helpfulness": None, "notes": "Invalid JSON", "raw": text}

    return data


def get_best_device():
    """Returns the best device on this computer"""

    if torch.cuda.is_available():
        device = torch.device("cuda")
        total_memory = torch.cuda.get_device_properties(device).total_memory
        print(f"GPU Memory: {total_memory / 1e9:.1f} GB")
        print(f"GPU Name: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Found device: {device}")
    return device

def get_text_from_index(
    index: pt.terrier.TerrierIndex, doc_ids: List[str]
) -> Dict[str, str]:
    """
    Retrieve document text from the index metadata.

    Args:
        index: PyTerrier index reference
        doc_ids: List of document IDs

    Returns:
        Dict mapping doc_id -> text
    """
    meta_index = index.meta_index()

    result = {}
    for doc_id in doc_ids:
        try:
            # Get internal docid from docno
            docid = meta_index.getDocument("docno", doc_id)
            if docid >= 0:
                text = meta_index.getItem("text", docid)
                result[doc_id] = text
        except Exception:
            pass  # Document not found

    return result


# Helper for single document lookup
def get_doc_text(index, doc_id: str) -> str:
    """Get text for a single document from the index."""
    texts = get_text_from_index(index, [doc_id])
    return texts.get(doc_id, "")


def doc_exists_in_index(index, doc_id: str) -> bool:
    """Check if a document exists in the index (without fetching text)."""
    meta_index = index.meta_index()
    try:
        return meta_index.getDocument("docno", doc_id) >= 0
    except Exception:
        return False

# Convert PyTerrier dataframes to ir-measures format
def to_ir_measures_qrels(qrels_df):
    """Convert PyTerrier qrels to ir-measures format."""
    return qrels_df.rename(
        columns={"qid": "query_id", "docno": "doc_id", "label": "relevance"}
    )


def to_ir_measures_run(run_df):
    """Convert PyTerrier run to ir-measures format."""
    return run_df.rename(columns={"qid": "query_id", "docno": "doc_id"})
    
def clean_query(text):
    # Replace hyphens and other special chars with a space
    # This keeps the words but removes the "operator" context
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    # Remove extra whitespace
    return " ".join(cleaned.split())

def find_hard_negatives_bm25(
    query: str,
    positive_doc_ids: set,
    retriever,
    num_negatives: int = 5,
) -> List[str]:
    """
    Find hard negatives using BM25.

    Hard negatives are documents that are lexically similar to the query
    but are NOT the ground truth positive document.

    Args:
        query: The query text
        positive_doc_ids: Set of positive document IDs (to exclude)
        retriever: BM25 retriever
        num_negatives: Number of hard negatives to return

    Returns:
        List of document IDs for hard negatives
    """
    # Implement hard negative mining with BM25

    # 1. Retrieve top documents with BM25
    results = retriever.search(clean_query(query))
    hard_negatives = []
    for _, row in results.iterrows():
        docno = row["docno"]
        # 2. Filter out the positive documents
        if docno not in positive_doc_ids:
            hard_negatives.append(docno)
            # 3. Return the top-k remaining as hard negatives
            if len(hard_negatives) >= num_negatives:
                break
    return hard_negatives

def create_training_triplets(
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    index,
    retriever,
    num_hard_negatives: int = 3,
    max_queries: int = None,
) -> List[Dict]:
    """
    Create training triplets with hard negatives.

    Each triplet contains:
    - query: The question
    - positive: A relevant document
    - negatives: List of hard negative documents

    Args:
        queries_df: DataFrame with queries
        qrels_df: DataFrame with relevance judgments
        index: PyTerrier index reference (for text retrieval)
        retriever: BM25 retriever for hard negative mining
        num_hard_negatives: Number of hard negatives per query
        max_queries: Maximum number of queries to process

    Returns:
        List of training triplets
    """
    triplets = []

    queries_to_process = queries_df.head(max_queries) if max_queries else queries_df

    for _, query_row in tqdm(
        queries_to_process.iterrows(),
        desc="Creating training triplets",
        total=len(queries_to_process),
    ):
        qid = query_row["qid"]
        query_text = query_row["query"]

        # Get positive documents for this query
        positive_doc_ids = set(qrels_df[qrels_df["qid"] == qid]["docno"].tolist())

        if not positive_doc_ids:
            continue

        # Get the first positive document
        positive_id = list(positive_doc_ids)[0]
        positive_text = get_doc_text(index, positive_id)

        if not positive_text:
            continue

        # Find hard negatives
        hard_neg_ids = find_hard_negatives_bm25(
            query_text, positive_doc_ids, retriever, num_hard_negatives
        )
        hard_neg_texts = get_text_from_index(index, hard_neg_ids)
        hard_neg_list = [
            hard_neg_texts[nid] for nid in hard_neg_ids if nid in hard_neg_texts
        ]

        if hard_neg_list:
            triplets.append(
                {
                    "query": query_text,
                    "positive": positive_text,
                    "negatives": hard_neg_list,
                }
            )

    return triplets

def contrastive_loss(
    query_rep: torch.Tensor,
    doc_rep: torch.Tensor,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Compute InfoNCE contrastive loss.

    Args:
        query_rep: (batch_size, vocab_size)
        doc_rep: (batch_size, vocab_size)
        temperature: Temperature for softmax

    Returns:
        Average loss
    """
    # Implement contrastive loss

    # 1. Normalize representations
    query_rep_norm = F.normalize(query_rep, p=2, dim=-1)
    doc_rep_norm = F.normalize(doc_rep, p=2, dim=-1)

    # 2. Compute similarity scores (dot product)
    # scores shape: (batch_size, batch_size)
    scores = torch.matmul(query_rep_norm, doc_rep_norm.transpose(0, 1)) / temperature

    # 3. Create labels for positive pairs (diagonal elements)
    # labels shape: (batch_size,)
    labels = torch.arange(len(scores)).to(scores.device)

    # 4. Compute InfoNCE loss
    # F.cross_entropy combines log_softmax and NLLLoss
    loss = F.cross_entropy(scores, labels)

    return loss

def train_splade(
    encoder: SPLADEEncoder,
    train_data: List[Dict],
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    temperature: float = 0.05,
    use_hard_negatives: bool = False,
):
    """
    Train the SPLADE encoder.

    Args:
        encoder: SPLADE model
        train_data: List of triplets {query, positive, negatives}
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        temperature: Temperature for InfoNCE
        use_hard_negatives: If True, expects triplets with 'negatives' field
    """
    from torch.optim import AdamW

    optimizer = AdamW(encoder.parameters(), lr=learning_rate)
    encoder.train()

    for epoch in range(num_epochs):
        total_loss = 0
        total_contrastive = 0
        total_flops = 0
        num_batches = 0

        # Shuffle data
        import random

        shuffled = train_data.copy()
        random.shuffle(shuffled)

        # Create batches
        for i in tqdm(range(0, len(shuffled), batch_size), desc=f"Epoch {epoch + 1}"):
            batch = shuffled[i : i + batch_size]
            if len(batch) < 2:  # Need at least 2 for contrastive
                continue

            if use_hard_negatives:
                # Use triplets with hard negatives
                queries = [ex["query"] for ex in batch]
                positives = [ex["positive"][:500] for ex in batch]
                # Collect hard negatives from all examples in batch
                all_negatives = []
                for ex in batch:
                    all_negatives.extend([n[:500] for n in ex.get("negatives", [])[:2]])

                # Encode queries
                query_rep = encoder.encode(queries)
                # Encode positives + negatives together
                all_docs = positives + all_negatives
                doc_reps = encoder.encode(all_docs)
                positive_rep = doc_reps[: len(positives)]
                negative_rep = doc_reps[len(positives) :]

                # Contrastive loss with in-batch + hard negatives
                cont_loss = contrastive_loss(query_rep, positive_rep, temperature)

                # Add hard negative loss if we have negatives
                if len(negative_rep) > 0:
                    # Compute scores with negatives and ensure they're lower
                    neg_scores = torch.matmul(
                        F.normalize(query_rep, p=2, dim=-1),
                        F.normalize(negative_rep, p=2, dim=-1).T,
                    )
                    # Margin loss: positive scores should be higher than negative
                    hard_neg_loss = F.relu(neg_scores.mean() + 0.2).mean()
                    cont_loss = cont_loss + 0.3 * hard_neg_loss

                flops_loss = encoder.compute_flops_loss(
                    query_rep
                ) + encoder.compute_flops_loss(doc_reps)
            else:
                # Standard in-batch negatives only
                queries = [ex["query"] for ex in batch]
                docs = [ex["positive"][:500] for ex in batch]

                # Forward
                query_rep, doc_rep, flops_loss = encoder(queries, docs)

                # Contrastive loss
                cont_loss = contrastive_loss(query_rep, doc_rep, temperature)

            # Total loss
            loss = cont_loss + flops_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_contrastive += cont_loss.item()
            total_flops += flops_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_cont = total_contrastive / num_batches
        avg_flops = total_flops / num_batches

        print(
            f"Epoch {epoch + 1}: Loss={avg_loss:.4f} (Contrastive={avg_cont:.4f}, FLOPS={avg_flops:.4f})"
        )

    encoder.eval()
    print("Training complete!")

class CrossEncoderTeacher(nn.Module):
    """
    Cross-encoder for scoring query-document pairs.

    Unlike the bi-encoder, the cross-encoder sees query and doc together,
    allowing richer interactions but preventing pre-computation.
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, 1)

    def forward(
        self,
        queries: List[str],
        documents: List[str],
    ) -> torch.Tensor:
        """
        Score query-document pairs.

        Args:
            queries: List of queries
            documents: List of documents (same length as queries)

        Returns:
            Relevance scores (batch_size,)
        """
        # Implement cross-encoder forward

        # 1. Concatenate query and document with [SEP]
        pairs = [q + " " + self.tokenizer.sep_token + " " + d for q, d in zip(queries, documents)]

        # 2. Tokenize
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        ).to(self.model.device)

        # 3. Get [CLS] embedding
        outputs = self.model(**inputs)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # 4. Classify to get score
        scores = self.classifier(cls_output)

        return scores.squeeze(-1)
def distillation_loss(
    student_scores: torch.Tensor,
    teacher_scores: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute distillation loss (MSE or KL divergence).

    Args:
        student_scores: Student (SPLADE) scores
        teacher_scores: Teacher (cross-encoder) scores
        temperature: Temperature for softening distributions (KL only)

    Returns:
        Distillation loss
    """
    # Simple MSE between scores
    return F.mse_loss(student_scores, teacher_scores.detach())

def train_splade_with_distillation(
    encoder: SPLADEEncoder,
    teacher: CrossEncoderTeacher,
    train_data: List[Dict],
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    temperature: float = 1.0, # For distillation loss
    contrastive_temperature: float = 0.05, # For InfoNCE loss
    alpha: float = 0.5, # Weight for distillation loss vs. InfoNCE
):
    """
    Train the SPLADE encoder using distillation from a cross-encoder teacher.

    Args:
        encoder: SPLADE model (student)
        teacher: Cross-encoder model (teacher)
        train_data: List of triplets {query, positive, negatives} (only query and positive are used for distillation)
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        temperature: Temperature for distillation loss
        contrastive_temperature: Temperature for InfoNCE loss
        alpha: Weight for distillation loss (1 - alpha for InfoNCE)
    """
    from torch.optim import AdamW

    optimizer = AdamW(encoder.parameters(), lr=learning_rate)
    encoder.train()
    teacher.eval() # Teacher should be in evaluation mode

    for epoch in range(num_epochs):
        total_loss = 0
        total_distillation_loss = 0
        total_contrastive_loss = 0
        total_flops_loss = 0
        num_batches = 0

        import random

        shuffled = train_data.copy()
        random.shuffle(shuffled)

        for i in tqdm(range(0, len(shuffled), batch_size), desc=f"Epoch {epoch + 1} (Distillation)"):
            batch = shuffled[i : i + batch_size]
            if len(batch) < 1: # We can do distillation with single example, though batching is better
                continue

            queries = [ex["query"] for ex in batch]
            positives = [ex["positive"][:500] for ex in batch]

            # --- Student (SPLADE) forward pass ---
            query_rep = encoder.encode(queries)
            positive_rep = encoder.encode(positives)

            # Compute SPLADE scores (dot product between query and positive reps)
            # We use an identity matrix as labels for contrastive loss, so diagonal elements are positives.
            # The scores are (batch_size, batch_size)
            splade_scores = torch.matmul(
                F.normalize(query_rep, p=2, dim=-1),
                F.normalize(positive_rep, p=2, dim=-1).transpose(0, 1)
            )

            # --- Teacher (Cross-Encoder) forward pass ---
            # Get teacher scores for all query-positive pairs in the batch
            teacher_q_p_pairs = []
            for q_text in queries:
                for p_text in positives:
                    teacher_q_p_pairs.append((q_text, p_text))

            teacher_queries = [p[0] for p in teacher_q_p_pairs]
            teacher_docs = [p[1] for p in teacher_q_p_pairs]

            with torch.no_grad(): # Teacher predictions are fixed
                all_teacher_scores = teacher(teacher_queries, teacher_docs)
            
            # Reshape teacher scores to (batch_size, batch_size) to match student scores
            teacher_scores_reshaped = all_teacher_scores.reshape(len(queries), len(positives))

            # --- Losses ---
            # 1. Distillation loss: Student trying to match teacher scores
            dist_loss = distillation_loss(
                splade_scores,
                teacher_scores_reshaped,
                temperature=temperature,
            )

            # 2. Contrastive loss: InfoNCE for positive pairs (in-batch negatives)
            cont_loss = contrastive_loss(query_rep, positive_rep, contrastive_temperature)

            # 3. FLOPS loss: Sparsity penalty
            flops_loss = encoder.compute_flops_loss(query_rep) + encoder.compute_flops_loss(positive_rep)

            # Combined loss
            loss = alpha * dist_loss + (1 - alpha) * cont_loss + flops_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_distillation_loss += dist_loss.item()
            total_contrastive_loss += cont_loss.item()
            total_flops_loss += flops_loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_dist_loss = total_distillation_loss / num_batches
        avg_cont_loss = total_contrastive_loss / num_batches
        avg_flops_loss = total_flops_loss / num_batches

        print(
            f"Epoch {epoch + 1}: Loss={avg_loss:.4f} (Distillation={avg_dist_loss:.4f}, Contrastive={avg_cont_loss:.4f}, FLOPS={avg_flops_loss:.4f})"
        )

    encoder.eval()
    print("Distillation training complete!")

def retriever_to_results(
    retriever_fn,
    queries_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a retriever function to a results DataFrame for ir-measures.

    Args:
        retriever_fn: Function that takes a query and returns [(doc_id, score)]
        queries_df: DataFrame with queries

    Returns:
        DataFrame with columns [qid, docno, score]
    """
    all_results = []

    for _, query_row in tqdm(
        queries_df.iterrows(), desc="Retrieving", total=len(queries_df)
    ):
        qid = query_row["qid"]
        query_text = query_row["query"]

        results = retriever_fn(query_text)
        for doc_id, score in results:
            all_results.append({"qid": qid, "docno": doc_id, "score": score})

    return pd.DataFrame(all_results)

class RAGPipeline:
    """
    Complete RAG pipeline: Retrieval + Generation.

    Uses PyTerrier index for document storage (scalable to large collections).
    """

    def __init__(
        self,
        retriever: SPLADEEncoder,
        generator,
        generator_tokenizer,
        pt_index,
        doc_ids: List[str],
        top_k: int = 3,
    ):
        super().__init__()
        self.retriever = retriever
        self.generator = generator
        self.gen_tokenizer = generator_tokenizer
        self.pt_index = pt_index
        self.doc_ids = doc_ids
        self.top_k = top_k

        # Pre-compute document representations
        self._index_documents()

    def _get_doc_text(self, doc_id: str) -> str:
        """Get document text from PyTerrier index."""
        return get_doc_text(self.pt_index, doc_id)

    def _get_texts(self, doc_ids: List[str]) -> Dict[str, str]:
        """Get multiple document texts from PyTerrier index."""
        return get_text_from_index(self.pt_index, doc_ids)

    def _index_documents(self, batch_size: int = 16):
        """Index all documents with SPLADE."""
        print("Indexing documents with SPLADE...")

        # Get texts from PyTerrier index in batches
        all_reps = []

        with torch.no_grad():
            for i in tqdm(range(0, len(self.doc_ids), batch_size)):
                batch_ids = self.doc_ids[i : i + batch_size]
                batch_texts = self._get_texts(batch_ids)
                # Truncate texts and maintain order
                texts = [batch_texts.get(doc_id, "")[:500] for doc_id in batch_ids]
                reps = self.retriever.encode(texts)
                all_reps.append(reps.cpu())

        self.doc_reps = torch.cat(all_reps, dim=0)
        print(f"Documents indexed: {len(self.doc_ids)}")

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Retrieve most relevant documents.

        Args:
            query: User question
            top_k: Number of documents to retrieve

        Returns:
            List of (doc_id, score)
        """
        if top_k is None:
            top_k = self.top_k

        # Encode query
        with torch.no_grad():
            query_rep = self.retriever.encode([query]).cpu()

        # Compute scores
        scores = torch.matmul(query_rep, self.doc_reps.T).squeeze(0)

        # Top-k
        top_indices = torch.topk(scores, k=min(top_k, len(scores))).indices

        results = []
        for idx in top_indices:
            doc_id = self.doc_ids[idx]
            score = scores[idx].item()
            results.append((doc_id, score))

        return results

    def generate(
        self,
        query: str,
        context: str,
        max_new_tokens: int = 200,
    ) -> str:
        """Generate a response based on context."""
        system_msg = "You are a helpful assistant that answers technical questions using the provided context."
        user_msg = f"Context:\n{context}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = self.gen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.gen_tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.generator.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.gen_tokenizer.eos_token_id,
            )

        response = self.gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the response
        if "assistant" in response.lower():
            response = response.split("assistant")[-1].strip()

        return response

    def __call__(self, query: str) -> Dict:
        """
        Complete pipeline: retrieval + generation.

        Args:
            query: User question

        Returns:
            Dict with retrieved_docs and generated_answer
        """
        # Implement RAG pipeline

        # 1. Retrieve top-k documents
        retrieved_docs = self.retrieve(query)

        # 2. Build context from documents
        doc_texts = self._get_texts([doc_id for doc_id, _ in retrieved_docs])
        context = "\n\n".join([doc_texts[doc_id][:300] for doc_id, _ in retrieved_docs])

        # 3. Generate response
        generated_answer = self.generate(query, context)

        return {"query": query, "retrieved_docs": retrieved_docs, "generated_answer": generated_answer}

def splade_retriever(query: str) -> List[Tuple[str, float]]:
    return rag.retrieve(query, top_k=20)