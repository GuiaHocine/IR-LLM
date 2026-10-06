"""Regression tests for experiment boundaries and generated query handling."""

import json
import subprocess
import sys

import pytest

from IR_training_evaluation import load_synthetic_pairs, split_queries
from synthetic_data_gen import select_training_documents, split_numbered_questions


@pytest.mark.parametrize("script", ["IR_training_evaluation.py", "synthetic_data_gen.py"])
def test_help_without_model_loading(script):
    result = subprocess.run([sys.executable, script, "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--index-path" in result.stdout


def test_query_split_is_reproducible_and_disjoint():
    pd = pytest.importorskip("pandas")
    topics = pd.DataFrame(
        {"qid": [str(i) for i in range(12)], "query": [f"query {i}" for i in range(12)]}
    )
    train, test = split_queries(topics, 3, seed=42)
    train_again, test_again = split_queries(topics, 3, seed=42)
    assert set(train.qid).isdisjoint(test.qid)
    assert set(train.qid) | set(test.qid) == set(topics.qid)
    assert train.equals(train_again) and test.equals(test_again)
    with pytest.raises(ValueError):
        split_queries(topics, len(topics))


def test_training_documents_exclude_nonrelevant_and_held_out():
    pd = pytest.importorskip("pandas")
    train_qrels = pd.DataFrame({"docno": ["train", "shared", "irrelevant"], "label": [1, 1, 0]})
    test_qrels = pd.DataFrame({"docno": ["shared", "other"], "label": [1, 1]})
    assert select_training_documents(train_qrels, test_qrels, 10) == ["train"]
    assert select_training_documents(
        train_qrels, test_qrels, 1, available_doc_ids=["train", "shared"]
    ) == ["train"]
    assert select_training_documents(train_qrels, test_qrels, 1, available_doc_ids=["shared"]) == []


def test_synthetic_pairs_exclude_test_queries_documents_and_duplicates(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(
        json.dumps(
            [
                {"query": " training query ", "doc_id": 4},
                {"query": "training query", "doc_id": "4"},
                {"query": "held-out document question", "doc_id": "5"},
                {"query": "Test Question", "doc_id": "6"},
            ]
        )
    )
    assert load_synthetic_pairs(path, ["5"], ["test question"]) == [
        {"query": "training query", "doc_id": "4"}
    ]


def test_invalid_pair_schema_is_reported(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text('[{"query": "question"}]')
    with pytest.raises(ValueError, match="doc_id"):
        load_synthetic_pairs(path)


def test_generated_questions_split_numbering_bullets_and_duplicates():
    assert split_numbered_questions(
        "Here are questions:\n1. First question?\n2) Second question?\n- First question?"
    ) == ["First question?", "Second question?"]
    assert split_numbered_questions("  ") == []


def test_generation_decodes_only_completion(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from synthetic_data_gen import generate_text

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))

    class Tensor:
        shape = (1, 3)

        def __getitem__(self, key):
            assert key == (0, slice(3, None))
            return [4, 5]

    class Inputs(dict):
        def to(self, device):
            assert device == "cpu"
            return self

    class Tokenizer:
        eos_token_id = 0

        def __call__(self, prompt, return_tensors, add_special_tokens):
            assert add_special_tokens is False
            return Inputs(input_ids=Tensor())

        def decode(self, tokens, skip_special_tokens):
            assert tokens == [4, 5]
            return "assistant configuration question?"

    class Model:
        device = "cpu"

        def generate(self, **kwargs):
            return Tensor()

    # The literal word "assistant" in generated content must not be stripped.
    assert generate_text(Model(), Tokenizer(), "prompt") == "assistant configuration question?"
