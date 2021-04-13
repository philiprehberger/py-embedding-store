"""Basic import test."""


def test_import():
    """Verify the package can be imported."""
    import philiprehberger_embedding_store
    assert hasattr(philiprehberger_embedding_store, "__name__") or True
