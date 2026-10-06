"""LoTTE dataset and Terrier index lifecycle."""

import json
from itertools import islice
from pathlib import Path


def load_dataset(dataset_id):
    """Load topics and relevance judgments from a PyTerrier dataset."""
    import pyterrier as pt

    dataset = pt.get_dataset(dataset_id)
    topics = dataset.get_topics().copy()
    qrels = dataset.get_qrels().copy()
    for frame, columns in ((topics, ("qid",)), (qrels, ("qid", "docno"))):
        for column in columns:
            frame[column] = frame[column].astype(str)
    return dataset, topics, qrels


def load_or_create_index(dataset, index_path, max_docs=None):
    """Reuse a text-bearing index or build one without deleting existing files.

    A document limit is useful for smoke runs, but changes the evaluation corpus.
    Reuse a separate path for each dataset and document limit.
    """
    import pyterrier as pt

    if max_docs is not None and max_docs < 1:
        raise ValueError("max_docs must be positive.")
    pt.java.init()
    index_path = Path(index_path).resolve()
    manifest_path = index_path / "ir_llm_index.json"
    dataset_id = getattr(dataset, "irds_ref", lambda: None)()
    expected = {"dataset": str(dataset_id), "max_docs": max_docs}
    if (index_path / "data.properties").exists():
        if manifest_path.exists():
            actual = json.loads(manifest_path.read_text(encoding="utf-8"))
            if actual != expected:
                raise ValueError(f"Index settings differ at {index_path}. Use a new --index-path.")
        index = pt.terrier.TerrierIndex(str(index_path))
        if "text" not in list(index.meta_index().getKeys()):
            raise ValueError("Existing index has no text metadata; use a new --index-path.")
        return index
    if index_path.exists() and any(index_path.iterdir()):
        raise ValueError(f"Index directory is nonempty: {index_path}. Use a new path.")
    index_path.mkdir(parents=True, exist_ok=True)
    corpus = dataset.get_corpus_iter()
    if max_docs is not None:
        corpus = islice(corpus, max_docs)
    indexer = pt.terrier.IterDictIndexer(
        str(index_path),
        meta={"docno": 128, "text": 8192},
        meta_reverse=["docno"],
    )
    indexer.index(corpus)
    manifest_path.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    return pt.terrier.TerrierIndex(str(index_path))


def index_document_ids(index):
    """Return all indexed document IDs, independent of relevance judgments."""
    meta = index.meta_index()
    count = index.collection_statistics().getNumberOfDocuments()
    return [str(meta.getItem("docno", i)) for i in range(count)]
