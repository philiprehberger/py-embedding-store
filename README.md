# philiprehberger-embedding-store

[![Tests](https://github.com/philiprehberger/py-embedding-store/actions/workflows/publish.yml/badge.svg)](https://github.com/philiprehberger/py-embedding-store/actions/workflows/publish.yml)
[![PyPI version](https://img.shields.io/pypi/v/philiprehberger-embedding-store.svg)](https://pypi.org/project/philiprehberger-embedding-store/)
[![Last updated](https://img.shields.io/github/last-commit/philiprehberger/py-embedding-store)](https://github.com/philiprehberger/py-embedding-store/commits/main)
In-memory vector store with cosine similarity search.

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

# Filter by metadata
results = store.search(query, top_k=10, filter=lambda m: m["category"] == "tech")

# Minimum score threshold
results = store.search(query, min_score=0.7)

# Persistence
store.save("vectors.json")
loaded = VectorStore.load("vectors.json")

# Batch operations
store.add_many([("id1", emb1, meta1), ("id2", emb2, meta2)])
```

## API

| Method | Description |
|--------|-------------|
| `add(id, embedding, metadata?)` | Add a vector |
| `add_many(items)` | Batch add |
| `search(query, top_k?, metric?, filter?, min_score?)` | Similarity search |
| `get(id)` | Get entry by ID |
| `delete(id)` | Delete entry |
| `update_metadata(id, metadata)` | Update metadata |
| `save(path)` | Save to JSON |
| `VectorStore.load(path)` | Load from JSON |
| `clear()` | Remove all entries |
| `ids()` | List all IDs |

## Distance Metrics

- `"cosine"` (default) — cosine similarity
- `"dot"` — dot product

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
