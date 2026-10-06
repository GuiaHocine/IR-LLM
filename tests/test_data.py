"""Test index selection without a JVM or a downloaded dataset."""

import json
import sys
import types
from unittest.mock import Mock, patch

import pytest

from ir_llm.data import index_document_ids, load_or_create_index


def fake_pyterrier():
    index = Mock()
    index.meta_index.return_value.getKeys.return_value = ["docno", "text"]
    return types.SimpleNamespace(
        java=types.SimpleNamespace(init=Mock()),
        terrier=types.SimpleNamespace(
            TerrierIndex=Mock(return_value=index), IterDictIndexer=Mock()
        ),
    )


def test_existing_index_rejects_changed_corpus_limit(tmp_path):
    (tmp_path / "data.properties").touch()
    (tmp_path / "ir_llm_index.json").write_text(
        json.dumps({"dataset": "collection", "max_docs": 5000})
    )
    dataset = Mock()
    dataset.irds_ref.return_value = "collection"
    with patch.dict(sys.modules, {"pyterrier": fake_pyterrier()}):
        with pytest.raises(ValueError, match="settings differ"):
            load_or_create_index(dataset, tmp_path, max_docs=10000)


def test_nonempty_directory_is_preserved(tmp_path):
    existing = tmp_path / "important.txt"
    existing.write_text("keep this file")
    with patch.dict(sys.modules, {"pyterrier": fake_pyterrier()}):
        with pytest.raises(ValueError, match="nonempty"):
            load_or_create_index(Mock(), tmp_path, max_docs=5)
    assert existing.read_text() == "keep this file"


def test_document_candidates_include_whole_index():
    index = Mock()
    index.collection_statistics.return_value.getNumberOfDocuments.return_value = 3
    index.meta_index.return_value.getItem.side_effect = lambda key, i: ["a", "b", "c"][i]
    assert index_document_ids(index) == ["a", "b", "c"]
