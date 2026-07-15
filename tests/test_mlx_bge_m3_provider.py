from __future__ import annotations

import os
import subprocess
import sys
import unittest

from kb.embeddings.base import EmbeddingResult
from kb.embeddings.bge_m3_provider import MlxBgeM3Provider, embed_joint_documents


class _Tokenizer:
    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"t{token_id}"


class _JointBackend:
    def __init__(self) -> None:
        self.calls = 0

    def embed_batch(self, texts):
        self.calls += 1
        return [EmbeddingResult([float(index)], {f"t{index}": 1.0}) for index, _ in enumerate(texts)]


class _Wrapper:
    def __init__(self, backend) -> None:
        self.backend = backend


class MlxBgeM3ProviderTests(unittest.TestCase):
    def test_sparse_aggregation_uses_positive_bias_semantics_and_max_per_token(self) -> None:
        provider = MlxBgeM3Provider.__new__(MlxBgeM3Provider)
        provider.special_ids = {0, 1, 2, 3}
        provider.sparse_top_k = 2
        provider.tokenizer = _Tokenizer()

        result = provider._aggregate_sparse([0, 7, 7, 8, 9], [50.0, 0.2, 0.8, 0.4, 0.1])

        self.assertEqual(list(result), ["t7", "t8"])
        self.assertEqual(result["t7"], 0.8)

    def test_joint_dispatch_calls_shared_backbone_once(self) -> None:
        backend = _JointBackend()
        dense, sparse = embed_joint_documents(_Wrapper(backend), _Wrapper(backend), ["a", "b", "c"])

        self.assertEqual(backend.calls, 1)
        self.assertEqual([row[0] for row in dense], [0.0, 1.0, 2.0])
        self.assertEqual(list(sparse[2]), ["t2"])

    def test_ptha_service_import_does_not_import_torch(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import sys, ptha.service; print('torch' in sys.modules)"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.stdout.strip(), "False")

    @unittest.skipUnless(os.environ.get("PTHA_MLX_MODEL_PATH"), "set PTHA_MLX_MODEL_PATH for local artifact smoke")
    def test_real_artifact_single_forward_and_original_order(self) -> None:
        provider = MlxBgeM3Provider(
            os.environ["PTHA_MLX_MODEL_PATH"],
            model_revision=os.environ.get("PTHA_MLX_MODEL_REVISION", "local-validation"),
            batch_size=4,
        )
        before = provider.forward_calls
        results = provider.embed_batch(["short", "a somewhat longer synthetic input", "технический текст"])
        self.assertEqual(provider.forward_calls - before, 1)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(len(result.dense) == 1024 for result in results))
        self.assertEqual(provider.last_batch_metrics["chunks"], 3)


if __name__ == "__main__":
    unittest.main()
