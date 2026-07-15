from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ptha.cli import main
from ptha.config import load_config
from ptha.errors import DatabaseExistsError
from ptha.importer import import_archive
from ptha.paths import PthaPaths


class PthaShellTests(unittest.TestCase):
    def test_init_is_idempotent_and_preserves_existing_config(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / "config.toml"
            self.assertEqual(main(["--config", str(config), "init"]), 0)
            original = config.read_text(encoding="utf-8")
            self.assertEqual(main(["--config", str(config), "init"]), 0)
            self.assertEqual(config.read_text(encoding="utf-8"), original)

    def test_environment_overrides_config(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / "config.toml"
            config.write_text("config_version = 1\n[paths]\ndatabase = '/config.db'\n", encoding="utf-8")
            with patch.dict(os.environ, {"PTHA_DB_PATH": str(Path(root) / "env.db")}):
                self.assertEqual(load_config(config).database, Path(root) / "env.db")

    def test_legacy_bge_m3_models_migrate_to_pinned_mlx_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / "config.toml"
            config.write_text(
                'config_version = 1\n[models]\ndense_model = "BAAI/bge-m3"\n'
                'sparse_model = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"\n',
                encoding="utf-8",
            )

            loaded = load_config(config)

            self.assertEqual(loaded.embedding_backend, "mlx")
            self.assertEqual(loaded.embedding_model, "anfedoro/bge-m3-mlx-fp16")
            self.assertEqual(loaded.embedding_model_revision, "58e70901dbba8de8f3df91b5a313bcefcb151bae")
            self.assertEqual(loaded.dense_model, loaded.embedding_model)
            self.assertEqual(loaded.sparse_model, loaded.embedding_model)

    def test_legacy_auto_runtime_settings_migrate_to_mlx_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            config = Path(root) / "config.toml"
            config.write_text(
                "config_version = 1\n[models]\n"
                'dense_model = "BAAI/bge-m3"\n'
                'sparse_model = "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1"\n'
                'dense_device = "auto"\nsparse_device = "auto"\n'
                'dense_dtype = "auto"\nsparse_dtype = "auto"\n',
                encoding="utf-8",
            )

            loaded = load_config(config)

            self.assertEqual(loaded.embedding_backend, "mlx")
            self.assertEqual(loaded.embedding_device, "gpu")
            self.assertEqual(loaded.embedding_dtype, "float16")
            self.assertEqual(loaded.dense_device, "gpu")
            self.assertEqual(loaded.sparse_device, "gpu")
            self.assertEqual(loaded.dense_dtype, "float16")
            self.assertEqual(loaded.sparse_dtype, "float16")
            self.assertEqual(loaded.batch_size, 4)

    def test_status_json_has_versioned_schema(self) -> None:
        with tempfile.TemporaryDirectory() as root, patch("sys.stdout") as stdout:
            config = Path(root) / "config.toml"
            main(["--config", str(config), "init"])
            stdout.reset_mock()
            self.assertEqual(main(["--config", str(config), "status", "--json"]), 0)
            payload = json.loads("".join(call.args[0] + "\n" for call in stdout.write.call_args_list))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["database"]["state"], "missing")

    def test_mcp_absolute_config_uses_selected_config_and_executable(self) -> None:
        with tempfile.TemporaryDirectory() as root, patch("sys.stdout") as stdout:
            config = Path(root) / "config.toml"
            with patch("ptha.cli.shutil.which", return_value="/opt/ptha/bin/ptha"):
                self.assertEqual(main(["--config", str(config), "mcp", "config", "--absolute"]), 0)
            payload = json.loads("".join(call.args[0] + "\n" for call in stdout.write.call_args_list))
            server = payload["mcpServers"]["ptha"]
            self.assertEqual(server["command"], "/opt/ptha/bin/ptha")
            self.assertEqual(server["args"], ["--config", str(config.resolve()), "mcp", "serve"])

    def test_import_refuses_to_replace_active_database(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            source = root_path / "export.zip"
            source.touch()
            config_file = root_path / "config.toml"
            database = root_path / "active.db"
            database.touch()
            config_file.write_text(f"config_version = 1\n[paths]\ndatabase = '{database}'\n", encoding="utf-8")
            with self.assertRaises(DatabaseExistsError):
                import_archive(source, load_config(config_file))

    def test_import_first_run_creates_config_without_explicit_init(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            paths = PthaPaths(root_path / "config", root_path / "data", root_path / "cache", root_path / "state",
                              root_path / "logs", root_path / "run")
            config_file = paths.config_file
            with patch("ptha.config.platform_paths", return_value=paths):
                self.assertEqual(main(["--config", str(config_file), "import", str(root_path / "missing.zip"), "--quiet"]), 6)
            self.assertTrue(config_file.is_file())
            self.assertTrue(paths.data_dir.is_dir())
            self.assertTrue(paths.runtime_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
