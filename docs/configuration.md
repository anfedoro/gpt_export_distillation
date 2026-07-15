# PTHA configuration

`ptha init` creates a versioned TOML configuration using platform-specific
defaults. Existing configuration is not overwritten.

The effective configuration can be selected with `--config PATH` or the
`PTHA_CONFIG` environment variable. `PTHA_DB_PATH` and
`PTHA_MODEL_CACHE_DIR` provide one-command path overrides.

The important production settings are:

```toml
[models]
embedding_backend = "mlx"
embedding_model = "anfedoro/bge-m3-mlx-fp16"
embedding_model_revision = "58e70901dbba8de8f3df91b5a313bcefcb151bae"
embedding_dtype = "float16"
embedding_device = "gpu"
embedding_batch_size = 4

[service]
startup_timeout_seconds = 120
shutdown_timeout_seconds = 30

[paths]
database = ""
working_dir = ""
model_cache = ""
```

Use `ptha doctor` after changing paths or model settings. PTHA fails before
indexing when the model repository, revision, dtype, tokenizer, sparse head, or
hidden dimension is incompatible.
