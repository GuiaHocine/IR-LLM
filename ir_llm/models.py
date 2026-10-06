"""Learned sparse encoder and pretrained cross-encoder teacher."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer


class SPLADEEncoder(nn.Module):
    """Masked-language-model logits pooled into sparse vocabulary weights."""

    def __init__(self, model_name="distilbert-base-uncased", sparsity_weight=0.0001):
        super().__init__()
        if sparsity_weight < 0:
            raise ValueError("sparsity_weight must be nonnegative")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.sparsity_weight = sparsity_weight
        self.vocab_size = self.model.config.vocab_size

    def encode(self, texts, max_length=256):
        if not texts:
            raise ValueError("texts must contain at least one document")
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(self.model.device)
        weights = torch.log1p(F.relu(self.model(**inputs).logits))
        weights = weights * inputs["attention_mask"].unsqueeze(-1)
        return weights.max(dim=1).values

    def compute_flops_loss(self, sparse_rep):
        """FLOPS regularization using mean activation, independent of batch size."""
        return self.sparsity_weight * sparse_rep.mean(dim=0).square().sum()

    def forward(self, query_texts, doc_texts):
        query_rep = self.encode(query_texts)
        doc_rep = self.encode(doc_texts)
        penalty = self.compute_flops_loss(query_rep) + self.compute_flops_loss(doc_rep)
        return query_rep, doc_rep, penalty


class CrossEncoderTeacher(nn.Module):
    """Score paired texts using a pretrained relevance classifier.

    Choose a checkpoint already fine-tuned for ranking; a base language model
    with a newly initialized classification head is not a trained teacher.
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6-v2"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        if self.model.config.num_labels != 1:
            raise ValueError("Teacher checkpoint must have a single relevance score output")

    def forward(self, queries, documents):
        if len(queries) != len(documents):
            raise ValueError("queries and documents must have matching lengths")
        inputs = self.tokenizer(
            queries, documents, padding=True, truncation=True, return_tensors="pt", max_length=256
        ).to(self.model.device)
        return self.model(**inputs).logits.squeeze(-1)
