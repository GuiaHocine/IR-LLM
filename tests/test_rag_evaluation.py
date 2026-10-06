"""Regressions for mislabeled thinking modes and mixed-up judge scores."""

import subprocess
import sys

from rag_test import evaluate_pipeline, summarize_scores


class FakePipeline:
    thinking = False

    def retrieve(self, query):
        return [("document", 1.0)]

    def _get_texts(self, doc_ids):
        return {"document": "Useful context"}

    def generate(self, query, context, max_new_tokens):
        assert context == "Useful context"
        return "reasoned answer" if self.thinking else "direct answer"


def test_thinking_modes_keep_answers_and_judgments_together():
    def judge(query, answer):
        rating = 5 if answer == "reasoned answer" else 2
        return {"relevance": rating, "helpfulness": rating}

    records = evaluate_pipeline(FakePipeline(), [{"qid": "q1", "query": "question"}], judge)
    assert records[0]["thinking"] is False
    assert records[0]["generated_answer"] == "direct answer"
    assert records[1]["thinking"] is True
    assert records[1]["generated_answer"] == "reasoned answer"
    summary = summarize_scores(records)
    assert summary["no_thinking"]["relevance"]["mean"] == 2
    assert summary["thinking"]["relevance"]["mean"] == 5


def test_invalid_judgments_are_counted_without_affecting_mean():
    records = [
        {"thinking": False, "judge": {"relevance": 4, "helpfulness": 3}},
        {"thinking": False, "judge": {"relevance": None, "helpfulness": True}},
        {"thinking": False, "judge": {"relevance": 9, "helpfulness": "5"}},
    ]
    summary = summarize_scores(records)
    assert summary["no_thinking"]["relevance"] == {"mean": 4, "valid": 1, "total": 3}
    assert summary["no_thinking"]["helpfulness"]["mean"] == 3
    assert summary["thinking"]["relevance"]["mean"] is None


def test_rag_help_works_without_loading_models():
    result = subprocess.run(
        [sys.executable, "rag_test.py", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "--queries-file" in result.stdout
