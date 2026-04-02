import json
import pytest
from philiprehberger_embedding_store import VectorStore, SearchResult


# --- Core operations ---


def test_add_and_size():
    store = VectorStore()
    store.add("a", [1.0, 0.0, 0.0])
    assert store.size == 1
    assert store.dimensions == 3


def test_add_sets_dimensions():
    store = VectorStore()
    store.add("a", [1.0, 2.0])
    assert store.dimensions == 2


def test_add_dimension_mismatch():
    store = VectorStore(dimensions=3)
    with pytest.raises(ValueError):
        store.add("a", [1.0, 2.0])


def test_add_many():
    store = VectorStore()
    store.add_many([
        ("a", [1.0, 0.0], {"label": "first"}),
        ("b", [0.0, 1.0], {"label": "second"}),
    ])
    assert store.size == 2


def test_add_many_without_metadata():
    store = VectorStore()
    store.add_many([("a", [1.0, 0.0]), ("b", [0.0, 1.0])])
    assert store.size == 2
    assert store.get("a").metadata == {}


def test_get():
    store = VectorStore()
    store.add("a", [1.0, 0.0], {"k": "v"})
    entry = store.get("a")
    assert entry is not None
    assert entry.id == "a"
    assert entry.metadata == {"k": "v"}


def test_get_missing():
    store = VectorStore()
    assert store.get("nope") is None


def test_delete():
    store = VectorStore()
    store.add("a", [1.0, 0.0])
    assert store.delete("a") is True
    assert store.size == 0
    assert store.delete("a") is False


def test_remove():
    store = VectorStore()
    store.add("a", [1.0, 0.0])
    assert store.remove("a") is True
    assert store.size == 0
    assert store.remove("a") is False


def test_remove_is_alias_for_delete():
    store = VectorStore()
    store.add("x", [1.0, 0.0])
    store.remove("x")
    assert "x" not in store


def test_update_metadata():
    store = VectorStore()
    store.add("a", [1.0, 0.0], {"old": True})
    assert store.update_metadata("a", {"new": True}) is True
    entry = store.get("a")
    assert entry.metadata["new"] is True
    assert entry.metadata["old"] is True


def test_update_metadata_missing():
    store = VectorStore()
    assert store.update_metadata("nope", {"x": 1}) is False


def test_ids():
    store = VectorStore()
    store.add("x", [1.0])
    store.add("y", [2.0])
    assert set(store.ids()) == {"x", "y"}


def test_clear():
    store = VectorStore()
    store.add("a", [1.0])
    store.clear()
    assert store.size == 0


def test_clear_resets_dimensions():
    store = VectorStore()
    store.add("a", [1.0, 0.0, 0.0])
    store.clear()
    assert store.size == 0


def test_len():
    store = VectorStore()
    assert len(store) == 0
    store.add("a", [1.0, 0.0])
    assert len(store) == 1


def test_contains():
    store = VectorStore()
    store.add("a", [1.0, 0.0])
    assert "a" in store
    assert "b" not in store


def test_invalid_dimensions():
    with pytest.raises(ValueError, match="positive"):
        VectorStore(dimensions=0)
    with pytest.raises(ValueError, match="positive"):
        VectorStore(dimensions=-1)


def test_id_overwrite():
    store = VectorStore()
    store.add("a", [1.0, 0.0], {"version": 1})
    store.add("a", [0.0, 1.0], {"version": 2})
    assert store.size == 1
    assert store.get("a").metadata["version"] == 2


def test_search_result_repr():
    r = SearchResult(id="test", score=0.9876, metadata={})
    assert "0.9876" in repr(r)


# --- Store-level metric ---


def test_store_default_metric():
    store = VectorStore()
    assert store.metric == "cosine"


def test_store_custom_metric():
    store = VectorStore(metric="euclidean")
    assert store.metric == "euclidean"


def test_store_invalid_metric():
    with pytest.raises(ValueError, match="Unknown metric"):
        VectorStore(metric="hamming")


# --- Search: cosine ---


def test_search_cosine():
    store = VectorStore()
    store.add("a", [1.0, 0.0], {"label": "a"})
    store.add("b", [0.0, 1.0], {"label": "b"})
    store.add("c", [0.7, 0.7], {"label": "c"})

    results = store.search([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].id == "a"
    assert results[0].score > results[1].score


def test_search_empty():
    store = VectorStore()
    assert store.search([1.0, 0.0]) == []


def test_search_with_filter():
    store = VectorStore()
    store.add("a", [1.0, 0.0], {"type": "good"})
    store.add("b", [0.9, 0.1], {"type": "bad"})

    results = store.search(
        [1.0, 0.0],
        filter=lambda m: m.get("type") == "good",
    )
    assert len(results) == 1
    assert results[0].id == "a"


def test_search_min_score():
    store = VectorStore()
    store.add("a", [1.0, 0.0])
    store.add("b", [0.0, 1.0])

    results = store.search([1.0, 0.0], min_score=0.5)
    assert all(r.score >= 0.5 for r in results)


def test_search_all_filtered_out():
    store = VectorStore()
    store.add("a", [1.0, 0.0], {"type": "bad"})
    results = store.search([1.0, 0.0], filter=lambda m: m.get("type") == "good")
    assert results == []


def test_search_top_k_limits_results():
    store = VectorStore()
    for i in range(10):
        store.add(f"v{i}", [float(i), 0.0])
    results = store.search([5.0, 0.0], top_k=3)
    assert len(results) == 3


# --- Search: dot product ---


def test_search_dot_metric():
    store = VectorStore()
    store.add("a", [1.0, 0.0])
    store.add("b", [0.0, 1.0])
    results = store.search([1.0, 0.0], metric="dot")
    assert results[0].id == "a"


def test_search_dot_store_level():
    store = VectorStore(metric="dot")
    store.add("a", [2.0, 0.0])
    store.add("b", [0.0, 3.0])
    results = store.search([1.0, 0.0])
    assert results[0].id == "a"
    assert results[0].score == pytest.approx(2.0)


# --- Search: euclidean ---


def test_search_euclidean():
    store = VectorStore()
    store.add("close", [1.0, 0.0])
    store.add("far", [10.0, 10.0])
    results = store.search([1.0, 0.1], metric="euclidean")
    assert results[0].id == "close"
    assert results[0].score > results[1].score


def test_search_euclidean_identical_vector():
    store = VectorStore()
    store.add("same", [3.0, 4.0])
    results = store.search([3.0, 4.0], metric="euclidean")
    assert results[0].score == pytest.approx(1.0)


def test_search_euclidean_store_level():
    store = VectorStore(metric="euclidean")
    store.add("a", [0.0, 0.0])
    store.add("b", [100.0, 100.0])
    results = store.search([0.0, 0.1])
    assert results[0].id == "a"


# --- Search: manhattan ---


def test_search_manhattan():
    store = VectorStore()
    store.add("close", [1.0, 0.0])
    store.add("far", [10.0, 10.0])
    results = store.search([1.0, 0.1], metric="manhattan")
    assert results[0].id == "close"
    assert results[0].score > results[1].score


def test_search_manhattan_identical_vector():
    store = VectorStore()
    store.add("same", [5.0, 5.0])
    results = store.search([5.0, 5.0], metric="manhattan")
    assert results[0].score == pytest.approx(1.0)


def test_search_manhattan_store_level():
    store = VectorStore(metric="manhattan")
    store.add("a", [0.0, 0.0])
    store.add("b", [100.0, 100.0])
    results = store.search([0.0, 0.1])
    assert results[0].id == "a"


# --- Search: metric override ---


def test_search_metric_override():
    store = VectorStore(metric="cosine")
    store.add("a", [2.0, 0.0])
    store.add("b", [0.0, 3.0])
    results = store.search([1.0, 0.0], metric="dot")
    assert results[0].id == "a"
    assert results[0].score == pytest.approx(2.0)


def test_search_unknown_metric():
    store = VectorStore()
    store.add("a", [1.0, 0.0])
    with pytest.raises(ValueError, match="Unknown metric"):
        store.search([1.0, 0.0], metric="hamming")


# --- Search: metadata filtering ---


def test_search_filter_by_category():
    store = VectorStore()
    store.add("d1", [1.0, 0.0], {"category": "docs"})
    store.add("d2", [0.9, 0.1], {"category": "code"})
    store.add("d3", [0.8, 0.2], {"category": "docs"})

    results = store.search(
        [1.0, 0.0],
        filter=lambda m: m.get("category") == "docs",
    )
    assert len(results) == 2
    assert all(r.metadata["category"] == "docs" for r in results)


def test_search_filter_with_multiple_conditions():
    store = VectorStore()
    store.add("d1", [1.0, 0.0], {"category": "docs", "lang": "en"})
    store.add("d2", [0.9, 0.1], {"category": "docs", "lang": "de"})
    store.add("d3", [0.8, 0.2], {"category": "code", "lang": "en"})

    results = store.search(
        [1.0, 0.0],
        filter=lambda m: m.get("category") == "docs" and m.get("lang") == "en",
    )
    assert len(results) == 1
    assert results[0].id == "d1"


# --- search_many ---


def test_search_many_basic():
    store = VectorStore()
    store.add("a", [1.0, 0.0])
    store.add("b", [0.0, 1.0])

    all_results = store.search_many(
        [[1.0, 0.0], [0.0, 1.0]],
        top_k=1,
    )
    assert len(all_results) == 2
    assert all_results[0][0].id == "a"
    assert all_results[1][0].id == "b"


def test_search_many_empty_store():
    store = VectorStore()
    all_results = store.search_many([[1.0, 0.0], [0.0, 1.0]])
    assert all_results == [[], []]


def test_search_many_with_filter():
    store = VectorStore()
    store.add("a", [1.0, 0.0], {"keep": True})
    store.add("b", [0.0, 1.0], {"keep": False})

    all_results = store.search_many(
        [[1.0, 0.0], [0.0, 1.0]],
        filter=lambda m: m.get("keep") is True,
    )
    assert len(all_results) == 2
    for results in all_results:
        assert len(results) == 1
        assert results[0].id == "a"


def test_search_many_with_min_score():
    store = VectorStore()
    store.add("a", [1.0, 0.0])
    store.add("b", [0.0, 1.0])

    all_results = store.search_many(
        [[1.0, 0.0], [0.0, 1.0]],
        min_score=0.99,
    )
    assert len(all_results) == 2
    assert all_results[0][0].id == "a"
    assert all_results[1][0].id == "b"


def test_search_many_with_metric():
    store = VectorStore()
    store.add("a", [2.0, 0.0])
    store.add("b", [0.0, 3.0])

    all_results = store.search_many(
        [[1.0, 0.0], [0.0, 1.0]],
        metric="dot",
        top_k=1,
    )
    assert all_results[0][0].id == "a"
    assert all_results[1][0].id == "b"


# --- Persistence ---


def test_save_and_load(tmp_path):
    store = VectorStore()
    store.add("a", [1.0, 2.0], {"k": "v"})
    store.add("b", [3.0, 4.0])

    path = tmp_path / "store.json"
    store.save(path)

    loaded = VectorStore.load(path)
    assert loaded.size == 2
    assert loaded.dimensions == 2
    entry = loaded.get("a")
    assert entry.metadata == {"k": "v"}


def test_save_and_load_preserves_metric(tmp_path):
    store = VectorStore(metric="euclidean")
    store.add("a", [1.0, 2.0])

    path = tmp_path / "store.json"
    store.save(path)

    loaded = VectorStore.load(path)
    assert loaded.metric == "euclidean"


def test_save_creates_valid_json(tmp_path):
    store = VectorStore()
    store.add("a", [1.0, 2.0], {"key": "value"})

    path = tmp_path / "store.json"
    store.save(path)

    data = json.loads(path.read_text())
    assert "dimensions" in data
    assert "metric" in data
    assert "entries" in data
    assert len(data["entries"]) == 1


def test_load_default_metric_when_missing(tmp_path):
    """Older JSON files without a metric field default to cosine."""
    path = tmp_path / "old.json"
    path.write_text(json.dumps({
        "dimensions": 2,
        "entries": [{"id": "a", "embedding": [1.0, 0.0], "metadata": {}}],
    }))
    loaded = VectorStore.load(path)
    assert loaded.metric == "cosine"
    assert loaded.size == 1


def test_roundtrip_search_after_load(tmp_path):
    store = VectorStore()
    store.add("a", [1.0, 0.0])
    store.add("b", [0.0, 1.0])

    path = tmp_path / "store.json"
    store.save(path)

    loaded = VectorStore.load(path)
    results = loaded.search([1.0, 0.0], top_k=1)
    assert results[0].id == "a"
