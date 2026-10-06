"""Training examples and sparse encoder optimization."""

import random

import torch
from torch.nn import functional as F
from torch.optim import AdamW

from .retrieval import find_hard_negatives_bm25, get_doc_text, get_text_from_index


def create_training_triplets(
    queries_df, qrels_df, index, retriever, num_hard_negatives=3, max_queries=None
):
    """Mine negatives and select a deterministic available relevant document.

    Only labels greater than zero count as positives. Known relevant documents
    are all excluded from negatives, including positives absent from the index.
    """
    if max_queries is not None and max_queries < 0:
        raise ValueError("max_queries must be nonnegative")
    positive_rows = qrels_df[qrels_df["label"] > 0].copy()
    positive_rows["qid"] = positive_rows["qid"].astype(str)
    positive_rows["docno"] = positive_rows["docno"].astype(str)
    positives_by_query = positive_rows.groupby("qid")["docno"].agg(set).to_dict()
    queries = queries_df.head(max_queries) if max_queries is not None else queries_df
    triplets = []
    for row in queries.itertuples(index=False):
        positives = positives_by_query.get(str(row.qid), set())
        positive_text = next(
            (text for doc_id in sorted(positives) if (text := get_doc_text(index, doc_id))), ""
        )
        if not positive_text:
            continue
        negative_ids = find_hard_negatives_bm25(row.query, positives, retriever, num_hard_negatives)
        texts = get_text_from_index(index, negative_ids)
        negatives = [texts[doc_id] for doc_id in negative_ids if texts.get(doc_id)]
        if negatives or num_hard_negatives == 0:
            triplets.append({"query": row.query, "positive": positive_text, "negatives": negatives})
    return triplets


def contrastive_loss(query_rep, doc_rep, temperature=0.05):
    """InfoNCE with corresponding positives first and optional extra negatives."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(query_rep) == 0 or len(doc_rep) < len(query_rep):
        raise ValueError("Each query needs a corresponding positive document")
    scores = F.normalize(query_rep, dim=-1) @ F.normalize(doc_rep, dim=-1).T / temperature
    labels = torch.arange(len(query_rep), device=scores.device)
    return F.cross_entropy(scores, labels)


def distillation_loss(student_scores, teacher_scores, temperature=1.0):
    """Match per-query teacher ranking distributions using KL divergence."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_scores.shape != teacher_scores.shape:
        raise ValueError("Student and teacher score shapes must match")
    return (
        F.kl_div(
            F.log_softmax(student_scores / temperature, dim=-1),
            F.softmax(teacher_scores.detach() / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )


def _validate_training(train_data, num_epochs, batch_size, learning_rate):
    if len(train_data) < 2:
        raise ValueError("Training requires at least two examples")
    if num_epochs < 1 or batch_size < 2 or learning_rate <= 0:
        raise ValueError("num_epochs >= 1, batch_size >= 2, and learning_rate > 0 are required")
    for example in train_data:
        if not example.get("query") or not example.get("positive"):
            raise ValueError("Each training example needs nonempty query and positive text")


def _batches(train_data, batch_size):
    shuffled = list(train_data)
    random.shuffle(shuffled)
    for start in range(0, len(shuffled), batch_size):
        batch = shuffled[start : start + batch_size]
        if len(batch) >= 2:
            yield batch


def train_splade(
    encoder,
    train_data,
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-5,
    temperature=0.05,
    use_hard_negatives=False,
):
    """Train with in-batch positives and optional mined negative documents."""
    _validate_training(train_data, num_epochs, batch_size, learning_rate)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    optimizer = AdamW(encoder.parameters(), lr=learning_rate)
    history = []
    encoder.train()
    for epoch in range(num_epochs):
        losses = []
        for batch in _batches(train_data, batch_size):
            queries = [example["query"] for example in batch]
            documents = [example["positive"] for example in batch]
            if use_hard_negatives:
                documents.extend(
                    negative for example in batch for negative in example.get("negatives", [])
                )
            query_rep, doc_rep, penalty = encoder(queries, documents)
            loss = contrastive_loss(query_rep, doc_rep, temperature) + penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        average = sum(losses) / len(losses)
        history.append(average)
        print(f"Epoch {epoch + 1}: loss={average:.4f}")
    encoder.eval()
    return history


def train_splade_with_distillation(
    encoder,
    teacher,
    train_data,
    num_epochs=10,
    batch_size=8,
    learning_rate=2e-5,
    temperature=1.0,
    contrastive_temperature=0.05,
    alpha=0.5,
):
    """Combine ranking distillation, InfoNCE, and sparsity regularization."""
    _validate_training(train_data, num_epochs, batch_size, learning_rate)
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    if temperature <= 0 or contrastive_temperature <= 0:
        raise ValueError("temperatures must be positive")
    optimizer = AdamW(encoder.parameters(), lr=learning_rate)
    encoder.train()
    teacher.eval()
    history = []
    for epoch in range(num_epochs):
        losses = []
        for batch in _batches(train_data, batch_size):
            queries = [example["query"] for example in batch]
            documents = [example["positive"] for example in batch]
            documents.extend(
                negative for example in batch for negative in example.get("negatives", [])
            )
            query_rep, doc_rep, penalty = encoder(queries, documents)
            student_scores = F.normalize(query_rep, dim=-1) @ F.normalize(doc_rep, dim=-1).T
            with torch.no_grad():
                teacher_scores = (
                    teacher(
                        [query for query in queries for _ in documents],
                        documents * len(queries),
                    )
                    .reshape(len(queries), len(documents))
                    .to(student_scores.device)
                )
            loss = (
                alpha * distillation_loss(student_scores, teacher_scores, temperature)
                + (1 - alpha) * contrastive_loss(query_rep, doc_rep, contrastive_temperature)
                + penalty
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        average = sum(losses) / len(losses)
        history.append(average)
        print(f"Epoch {epoch + 1} (distillation): loss={average:.4f}")
    encoder.eval()
    return history
