import json
import pytest
from philiprehberger_embedding_store import VectorStore, SearchResult


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


def test_search_dot_metric():
    store = VectorStore()
    store.add("a", [1.0, 0.0])
    store.add("b", [0.0, 1.0])
    results = store.search([1.0, 0.0], metric="dot")
    assert results[0].id == "a"


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


def test_search_result_repr():
    r = SearchResult(id="test", score=0.9876, metadata={})
    assert "0.9876" in repr(r)


# --- New tests for v0.2.0 ---


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


def test_search_unknown_metric():
    store = VectorStore()
    store.add("a", [1.0, 0.0])
    with pytest.raises(ValueError, match="Unknown metric"):
        store.search([1.0, 0.0], metric="euclidean")


def test_search_all_filtered_out():
    store = VectorStore()
    store.add("a", [1.0, 0.0], {"type": "bad"})
    results = store.search([1.0, 0.0], filter=lambda m: m.get("type") == "good")
    assert results == []


def test_add_many_without_metadata():
    store = VectorStore()
    store.add_many([("a", [1.0, 0.0]), ("b", [0.0, 1.0])])
    assert store.size == 2
    assert store.get("a").metadata == {}


def test_search_top_k_limits_results():
    store = VectorStore()
    for i in range(10):
        store.add(f"v{i}", [float(i), 0.0])
    results = store.search([5.0, 0.0], top_k=3)
    assert len(results) == 3


def test_clear_resets_dimensions():
    store = VectorStore()
    store.add("a", [1.0, 0.0, 0.0])
    store.clear()
    assert store.size == 0
