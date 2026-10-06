"""Compatibility imports for older notebooks; new code uses :mod:`ir_llm`."""

from ir_llm.models import CrossEncoderTeacher, SPLADEEncoder
from ir_llm.rag import (
    RAGPipeline,
    RAGPipelineBM25,
    RAGPipelineSplade,
    judge_answer_quality,
    strip_think,
)
from ir_llm.retrieval import (
    SPLADERetriever,
    clean_query,
    doc_exists_in_index,
    find_hard_negatives_bm25,
    get_doc_text,
    get_text_from_index,
    retriever_to_results,
    to_ir_measures_qrels,
    to_ir_measures_run,
)
from ir_llm.runtime import get_best_device
from ir_llm.training import (
    contrastive_loss,
    create_training_triplets,
    distillation_loss,
    train_splade,
    train_splade_with_distillation,
)

__all__ = [
    "CrossEncoderTeacher",
    "SPLADEEncoder",
    "RAGPipeline",
    "RAGPipelineBM25",
    "RAGPipelineSplade",
    "judge_answer_quality",
    "strip_think",
    "SPLADERetriever",
    "clean_query",
    "doc_exists_in_index",
    "find_hard_negatives_bm25",
    "get_doc_text",
    "get_text_from_index",
    "retriever_to_results",
    "to_ir_measures_qrels",
    "to_ir_measures_run",
    "get_best_device",
    "contrastive_loss",
    "create_training_triplets",
    "distillation_loss",
    "train_splade",
    "train_splade_with_distillation",
]
