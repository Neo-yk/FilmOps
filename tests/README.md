# Tests

Run with:

```bash
pip install -e ".[dev]"
pytest tests/
```

Layout:

* `tests/unit/` — unit tests that don't need real weights or videos.
* `tests/integration/` — smoke tests; some require checkpoints / GPU
  (marked with `@pytest.mark.requires_ckpt`).
* `tests/fixtures/` — small sample data.
