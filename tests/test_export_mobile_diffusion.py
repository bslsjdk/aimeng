import gzip
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.export_mobile_diffusion import export_checkpoint, pack_aimg


class MobileDiffusionExportTests(unittest.TestCase):
    def test_exports_json_and_android_aimg_v1(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / "checkpoint.pt"
            json_path = root / "mobile_diffusion.json"
            aimg_path = root / "mobile_diffusion.aimg"
            torch.save({
                "format": "aimeng-sparse-diffusion-char-v1",
                "config": {
                    "neurons": 8, "width": 4, "active_k": 2,
                    "fanout": 2, "max_steps": 2, "context": 4,
                },
                "stoi": {"<unk>": 0, "你": 1},
                "itos": ["<unk>", "你"],
                "model_state": {"embedding.weight": torch.tensor([[0., 0., 0., 0.], [1., 2., 3., 4.]])},
                "source": "user_corpus",
            }, checkpoint)

            result = export_checkpoint(checkpoint, json_path)
            packed_result = pack_aimg(json_path, aimg_path)

            self.assertEqual(result["format"], "aimeng-mobile-diffusion-json-v1")
            self.assertEqual(packed_result["format"], "AIMG")
            packed = aimg_path.read_bytes()
            self.assertEqual(packed[:4], b"AIMG")
            version, expected_length = struct.unpack(">II", packed[4:12])
            self.assertEqual(version, 1)
            raw = gzip.decompress(packed[44:])
            self.assertEqual(len(raw), expected_length)
            self.assertEqual(hashlib.sha256(raw).digest(), packed[12:44])
            self.assertEqual(json.loads(raw)["format"], "aimeng-mobile-diffusion-json-v1")


if __name__ == "__main__":
    unittest.main()
