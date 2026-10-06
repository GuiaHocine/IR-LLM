"""Document metadata, ranking adapters, and candidate retrieval."""

import re


def get_text_from_index(index, doc_ids):
    """Fetch stored document text, omitting IDs absent from the index.

    Metadata errors propagate so a broken index is not mistaken for missing data.
    """
    metadata = index.meta_index()
    texts = {}
    for doc_id in doc_ids:
        internal_id = metadata.getDocument("docno", str(doc_id))
        if internal_id >= 0:
            texts[doc_id] = metadata.getItem("text", internal_id)
    return texts


def get_doc_text(index, doc_id):
    """Return a document's text, or an empty string if it is absent."""
    return get_text_from_index(index, [doc_id]).get(doc_id, "")


def doc_exists_in_index(index, doc_id):
    return index.meta_index().getDocument("docno", str(doc_id)) >= 0


def clean_query(text):
    """Remove query parser operators and normalize whitespace."""
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def to_ir_measures_qrels(qrels_df):
    return qrels_df.rename(columns={"qid": "query_id", "docno": "doc_id", "label": "relevance"})


def to_ir_measures_run(run_df):
    return run_df.rename(columns={"qid": "query_id", "docno": "doc_id"})


def find_hard_negatives_bm25(query, positive_doc_ids, retriever, num_negatives=5):
    """Return distinct ranked documents that are not known positives."""
    if num_negatives < 0:
        raise ValueError("num_negatives must be nonnegative")
    if num_negatives == 0:
        return []
    positives = {str(doc_id) for doc_id in positive_doc_ids}
    negatives = []
    for doc_id in retriever.search(clean_query(query))["docno"]:
        doc_id = str(doc_id)
        if doc_id not in positives and doc_id not in negatives:
            negatives.append(doc_id)
            if len(negatives) == num_negatives:
                break
    return negatives


def retriever_to_results(retriever_fn, queries_df):
    """Adapt a text-to-ranked-documents callable to a PyTerrier run."""
    import pandas as pd

    rows = []
    for row in queries_df.itertuples(index=False):
        for doc_id, score in retriever_fn(row.query):
            rows.append({"qid": str(row.qid), "docno": str(doc_id), "score": float(score)})
    return pd.DataFrame(rows, columns=["qid", "docno", "score"])


class SPLADERetriever:
    """Exact dot-product retrieval over a small, in-memory candidate collection.

    Document representations are stored as sparse COO batches on CPU. Exact
    scoring still scans every candidate; this is not an inverted search index.
    """

    def __init__(self, retriever, pt_index, doc_ids, top_k=3, batch_size=16):
        if top_k < 0 or batch_size < 1:
            raise ValueError("top_k must be nonnegative and batch_size must be positive")
        self.retriever = retriever
        self.pt_index = pt_index
        self.doc_ids = list(dict.fromkeys(doc_ids))
        self.top_k = top_k
        self._index_documents(batch_size)

    def _get_doc_text(self, doc_id):
        return get_doc_text(self.pt_index, doc_id)

    def _get_texts(self, doc_ids):
        return get_text_from_index(self.pt_index, doc_ids)

    def _index_documents(self, batch_size=16):
        import torch

        representations = []
        was_training = self.retriever.training
        self.retriever.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(self.doc_ids), batch_size):
                    ids = self.doc_ids[start : start + batch_size]
                    texts = self._get_texts(ids)
                    dense = self.retriever.encode([texts.get(doc_id, "") for doc_id in ids]).cpu()
                    representations.append(dense.to_sparse().coalesce())
        finally:
            self.retriever.train(was_training)
        self.doc_reps = representations

    def retrieve(self, query, top_k=None):
        import torch

        top_k = self.top_k if top_k is None else top_k
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        if not self.doc_ids or top_k == 0:
            return []
        was_training = self.retriever.training
        self.retriever.eval()
        try:
            with torch.no_grad():
                query_rep = self.retriever.encode([query]).cpu()
        finally:
            self.retriever.train(was_training)
        scores = torch.cat(
            [torch.sparse.mm(batch, query_rep.T).squeeze(1) for batch in self.doc_reps]
        )
        indices = torch.topk(scores, k=min(top_k, len(self.doc_ids))).indices.tolist()
        return [(self.doc_ids[i], scores[i].item()) for i in indices]
