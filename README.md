# philiprehberger-embedding-store

[![Tests](https://github.com/philiprehberger/py-embedding-store/actions/workflows/publish.yml/badge.svg)](https://github.com/philiprehberger/py-embedding-store/actions/workflows/publish.yml)
[![PyPI version](https://img.shields.io/pypi/v/philiprehberger-embedding-store.svg)](https://pypi.org/project/philiprehberger-embedding-store/)
[![Last updated](https://img.shields.io/github/last-commit/philiprehberger/py-embedding-store)](https://github.com/philiprehberger/py-embedding-store/commits/main)

In-memory vector store with multi-metric similarity search.

## Installation

```bash
pip install philiprehberger-embedding-store
```

## Usage

```python
from philiprehberger_embedding_store import VectorStore

store = VectorStore(dimensions=1536)

# Add vectors with metadata
store.add("doc1", embedding=[0.1, 0.2, ...], metadata={"title": "First doc"})
store.add("doc2", embedding=[0.3, 0.1, ...], metadata={"title": "Second doc"})

# Search by similarity
results = store.search(query_embedding=[0.15, 0.18, ...], top_k=5)
for result in results:
    print(f"{result.id}: score={result.score:.3f}, {result.metadata}")
```

### Distance metrics

Choose a metric per store or override per search call:

```python
from philiprehberger_embedding_store import VectorStore

# Set default metric at store level
store = VectorStore(dimensions=128, metric="euclidean")
results = store.search(query, top_k=5)

# Override metric for a single search
results = store.search(query, top_k=5, metric="manhattan")
```

Supported metrics: `"cosine"` (default), `"dot"`, `"euclidean"`, `"manhattan"`.

### Metadata filtering

```python
from philiprehberger_embedding_store import VectorStore

store = VectorStore()
store.add("d1", [1.0, 0.0], {"category": "docs", "lang": "en"})
store.add("d2", [0.9, 0.1], {"category": "code", "lang": "en"})

# Filter by single field
results = store.search(query, filter=lambda m: m["category"] == "docs")

# Filter by multiple conditions
results = store.search(
    query,
    filter=lambda m: m["category"] == "docs" and m["lang"] == "en",
)
```

### Batch operations

```python
from philiprehberger_embedding_store import VectorStore

store = VectorStore()

# Add many vectors at once
store.add_many([
    ("id1", [0.1, 0.2], {"label": "first"}),
    ("id2", [0.3, 0.4], {"label": "second"}),
])

# Search with multiple queries at once
all_results = store.search_many(
    [query_embedding_1, query_embedding_2],
    top_k=5,
)
```

### Persistence

```python
from philiprehberger_embedding_store import VectorStore

store = VectorStore()
store.add("doc1", [0.1, 0.2], {"title": "Example"})

# Save to disk
store.save("vectors.json")

# Load from disk
loaded = VectorStore.load("vectors.json")
```

### Store management

```python
from philiprehberger_embedding_store import VectorStore

store = VectorStore()
store.add("a", [1.0, 0.0])

store.remove("a")      # Remove by ID
store.clear()           # Remove all entries
```

## API

| Function / Class | Description |
|------------------|-------------|
| `VectorStore(dimensions, metric?)` | Create a store with optional dimensionality and metric |
| `add(id, embedding, metadata?)` | Add a vector with optional metadata |
| `add_many(items)` | Batch add multiple vectors |
| `search(query, top_k?, metric?, filter?, min_score?)` | Similarity search |
| `search_many(queries, top_k?, metric?, filter?, min_score?)` | Batch similarity search |
| `get(id)` | Get entry by ID |
| `delete(id)` | Delete entry by ID |
| `remove(id)` | Remove entry by ID (alias for delete) |
| `update_metadata(id, metadata)` | Update metadata for an entry |
| `save(path)` | Save store to JSON file |
| `VectorStore.load(path)` | Load store from JSON file |
| `clear()` | Remove all entries |
| `ids()` | List all stored IDs |
| `len(store)` | Number of entries |
| `id in store` | Check if ID exists |
| `store.size` | Number of entries (property) |
| `store.metric` | Current distance metric (property) |

## Development

```bash
pip install -e .
python -m pytest tests/ -v
```

## Support

If you find this project useful:

⭐ [Star the repo](https://github.com/philiprehberger/py-embedding-store)

🐛 [Report issues](https://github.com/philiprehberger/py-embedding-store/issues?q=is%3Aissue+is%3Aopen+label%3Abug)

💡 [Suggest features](https://github.com/philiprehberger/py-embedding-store/issues?q=is%3Aissue+is%3Aopen+label%3Aenhancement)

❤️ [Sponsor development](https://github.com/sponsors/philiprehberger)

🌐 [All Open Source Projects](https://philiprehberger.com/open-source-packages)

💻 [GitHub Profile](https://github.com/philiprehberger)

🔗 [LinkedIn Profile](https://www.linkedin.com/in/philiprehberger)

## License

[MIT](LICENSE)
