"""Regression tests that do not download models or initialize Java."""

import contextlib
import sys
import types
import unittest
from unittest.mock import patch

from ir_llm.rag import generate_completion, parse_judgment, strip_think
from ir_llm.retrieval import (
    clean_query,
    find_hard_negatives_bm25,
    get_doc_text,
    get_text_from_index,
)


class FakeMetadata:
    def getDocument(self, field, doc_id):
        return {"a": 0, "b": 1}.get(doc_id, -1)

    def getItem(self, field, internal_id):
        return ["first document", "second document"][internal_id]


class FakeIndex:
    def meta_index(self):
        return FakeMetadata()


class RetrievalTests(unittest.TestCase):
    def test_metadata_preserves_ids_and_omits_missing_documents(self):
        self.assertEqual(
            get_text_from_index(FakeIndex(), ["b", "missing", "a"]),
            {"b": "second document", "a": "first document"},
        )
        self.assertEqual(get_doc_text(FakeIndex(), "missing"), "")

    def test_index_errors_are_not_silently_swallowed(self):
        class BrokenIndex:
            def meta_index(self):
                raise RuntimeError("corrupt metadata")

        with self.assertRaisesRegex(RuntimeError, "corrupt"):
            get_doc_text(BrokenIndex(), "a")

    def test_clean_query_preserves_unicode_words(self):
        self.assertEqual(clean_query("café + (disk-error)  !"), "café disk error")

    def test_hard_negatives_exclude_positives_and_duplicates(self):
        class Retriever:
            def search(self, query):
                self.query = query
                return {"docno": ["positive", "n1", "n1", "n2", "n3"]}

        retriever = Retriever()
        self.assertEqual(
            find_hard_negatives_bm25("disk-error", {"positive"}, retriever, 2), ["n1", "n2"]
        )
        self.assertEqual(retriever.query, "disk error")
        self.assertEqual(find_hard_negatives_bm25("query", set(), retriever, 0), [])

    def test_invalid_negative_count(self):
        with self.assertRaises(ValueError):
            find_hard_negatives_bm25("query", set(), None, -1)


class JudgeTests(unittest.TestCase):
    def test_extracts_valid_json_after_reasoning_and_code_fence(self):
        result = parse_judgment(
            '<think>{"relevance": 1}</think>```json\n{"relevance":4,"helpfulness":5,"notes":"useful"}\n```'
        )
        self.assertEqual(result, {"relevance": 4, "helpfulness": 5, "notes": "useful"})

    def test_rejects_out_of_range_and_boolean_scores(self):
        for score in (0, 6, True, "4", None):
            import json

            result = parse_judgment(
                json.dumps({"relevance": score, "helpfulness": 4, "notes": "example"})
            )
            self.assertIsNone(result["relevance"])

    def test_assistant_word_is_kept(self):
        self.assertEqual(
            strip_think("The assistant helps with indexing."), "The assistant helps with indexing."
        )

    def test_generation_decodes_only_new_tokens(self):
        class Inputs(dict):
            def to(self, device):
                return self

        class Outputs:
            def __getitem__(self, key):
                if key != (0, slice(3, None)):
                    raise AssertionError(f"Wrong completion slice: {key}")
                return [10, 11]

        class Tokenizer:
            eos_token_id = 9

            def apply_chat_template(self, messages, **kwargs):
                return "prompt containing assistant and a JSON example"

            def __call__(self, prompt, **kwargs):
                return Inputs(input_ids=types.SimpleNamespace(shape=(1, 3)))

            def decode(self, tokens, **kwargs):
                if tokens != [10, 11]:
                    raise AssertionError("Prompt was included in decoded text")
                return "The assistant answers correctly."

        class Model:
            device = "cpu"
            training = True

            def eval(self):
                self.training = False

            def train(self, mode):
                self.training = mode

            def generate(self, **kwargs):
                if "temperature" in kwargs:
                    raise AssertionError("Deterministic generation should not pass temperature")
                return Outputs()

        model = Model()
        fake_torch = types.SimpleNamespace(no_grad=contextlib.nullcontext)
        with patch.dict(sys.modules, {"torch": fake_torch}):
            answer = generate_completion(model, Tokenizer(), [], do_sample=False)
        self.assertEqual(answer, "The assistant answers correctly.")
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
