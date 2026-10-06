"""Numerical regression tests; optional locally, run in the ML CI environment."""

import importlib.util
import unittest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TrainingTests(unittest.TestCase):
    def test_hard_negatives_participate_in_softmax(self):
        import torch

        from ir_llm.training import contrastive_loss

        query = torch.eye(2)
        positives = torch.eye(2)
        baseline = contrastive_loss(query, positives, temperature=1)
        with_negative = contrastive_loss(
            query, torch.cat([positives, positives[:1]]), temperature=1
        )
        self.assertGreater(with_negative.item(), baseline.item())

    def test_distillation_has_gradients_for_student_only(self):
        import torch

        from ir_llm.training import distillation_loss

        student = torch.tensor([[0.0, 1.0]], requires_grad=True)
        teacher = torch.tensor([[1.0, 0.0]], requires_grad=True)
        loss = distillation_loss(student, teacher, temperature=2)
        loss.backward()
        self.assertGreater(loss.item(), 0)
        self.assertIsNotNone(student.grad)
        self.assertIsNone(teacher.grad)
        self.assertAlmostEqual(distillation_loss(teacher, teacher).item(), 0, places=6)

    def test_sparse_retriever_matches_dense_scores_and_restores_training(self):
        import torch

        from ir_llm.retrieval import SPLADERetriever

        class Metadata:
            def getDocument(self, key, doc_id):
                return {"a": 0, "b": 1}.get(doc_id, -1)

            def getItem(self, key, doc_id):
                return ["first", "second"][doc_id]

        class Index:
            def meta_index(self):
                return Metadata()

        class Encoder(torch.nn.Module):
            def encode(self, texts):
                vectors = {
                    "first": [2.0, 0.0, 0.0],
                    "second": [0.0, 1.0, 0.0],
                    "question": [1.0, 1.0, 0.0],
                }
                return torch.tensor([vectors[text] for text in texts])

        encoder = Encoder()
        retriever = SPLADERetriever(encoder, Index(), ["a", "b"], top_k=2, batch_size=1)
        self.assertTrue(all(batch.is_sparse for batch in retriever.doc_reps))
        self.assertEqual(retriever.retrieve("question"), [("a", 2.0), ("b", 1.0)])
        self.assertTrue(encoder.training)
        self.assertEqual(retriever.retrieve("question", top_k=0), [])
        self.assertEqual(SPLADERetriever(encoder, Index(), []).retrieve("question"), [])

    @unittest.skipUnless(importlib.util.find_spec("pandas") is not None, "pandas is not installed")
    def test_triplets_ignore_nonrelevant_qrels_and_zero_query_limit(self):
        import pandas as pd

        from ir_llm.training import create_training_triplets

        class Metadata:
            def getDocument(self, key, doc_id):
                return {"p": 0, "n": 1}.get(doc_id, -1)

            def getItem(self, key, doc_id):
                return ["relevant text", "negative text"][doc_id]

        class Index:
            def meta_index(self):
                return Metadata()

        class Retriever:
            def search(self, query):
                return {"docno": ["p", "n"]}

        queries = pd.DataFrame([{"qid": "1", "query": "question"}])
        qrels = pd.DataFrame(
            [
                {"qid": "1", "docno": "p", "label": 1},
                {"qid": "1", "docno": "n", "label": 0},
            ]
        )
        examples = create_training_triplets(
            queries, qrels, Index(), Retriever(), num_hard_negatives=1
        )
        self.assertEqual(
            examples,
            [{"query": "question", "positive": "relevant text", "negatives": ["negative text"]}],
        )
        self.assertEqual(
            create_training_triplets(queries, qrels, Index(), Retriever(), max_queries=0), []
        )

    def test_too_small_training_data_fails_before_optimization(self):
        from ir_llm.training import train_splade

        with self.assertRaisesRegex(ValueError, "at least two"):
            train_splade(None, [], num_epochs=1)


if __name__ == "__main__":
    unittest.main()


def test_distillation_trains_on_mined_negatives():
    import torch

    from ir_llm.training import train_splade_with_distillation

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, queries, documents):
            vectors = {"positive-a": [1.0, 0.0], "positive-b": [0.0, 1.0], "negative": [1.0, 1.0]}
            return (
                torch.eye(len(queries)) * self.weight,
                torch.tensor([vectors[text] for text in documents]) * self.weight,
                self.weight.square() * 0.001,
            )

    class Teacher(torch.nn.Module):
        def forward(self, queries, documents):
            assert len(queries) == len(documents) == 8
            assert documents.count("negative") == 4
            return torch.arange(len(documents), dtype=torch.float32)

    encoder = Encoder()
    history = train_splade_with_distillation(
        encoder,
        Teacher(),
        [
            {"query": "a", "positive": "positive-a", "negatives": ["negative"]},
            {"query": "b", "positive": "positive-b", "negatives": ["negative"]},
        ],
        num_epochs=1,
        batch_size=2,
    )
    assert len(history) == 1 and history[0] > 0
    assert encoder.weight.item() != 1.0
