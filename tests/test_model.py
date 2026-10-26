import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from bcf6.model import BCF6, halfcode_to_float, float_to_halfcode
from bcf6.deployment import export_model, load_model, replay_check
from bcf6.codec import (
    independent_decode,
    pack_blocks,
    decode_parameters,
    blocks_to_image,
)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_half_mapping_and_derivative(self):
        rng = np.random.default_rng(27)
        codes = rng.integers(-31743, 31744, size=2000, dtype=np.int32)
        bits = (np.abs(codes) | np.where(codes < 0, 32768, 0)).astype(np.uint16)
        expected = bits.view(np.float16).astype(np.float32)
        actual = halfcode_to_float(torch.tensor(codes, dtype=torch.float32)).numpy()
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(
            float_to_halfcode(torch.tensor(expected)).numpy(), codes, atol=0.01
        )
        v = torch.tensor(
            [12345.2, -14532.3, 0.25], dtype=torch.float64, requires_grad=True
        )
        self.assertTrue(
            torch.autograd.gradcheck(halfcode_to_float, (v,), eps=1e-4, atol=1e-6)
        )

    def test_training_export_reload_and_gradients(self):
        torch.manual_seed(3)
        model = BCF6("0.5K", 8, (32, 32), sides=[16, 8, 4, 4])
        uv = torch.rand(96, 2)
        target = torch.cat([uv, uv, uv, uv], -1)
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(3):
            opt.zero_grad()
            loss = (model(uv, 0) - target).square().mean()
            loss.backward()
            opt.step()
        model = model.to_blocks()
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        opt.zero_grad()
        loss = (model(uv, 0) - target).square().mean()
        loss.backward()
        active = model.pyramids[0][0]
        self.assertGreater(float(active.endpoints.grad.abs().sum()), 0)
        self.assertGreater(float(active.indices.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(active.endpoints.grad).all())
        opt.step()
        model.constrain()
        model.freeze_quantized()
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "saved"
            meta = export_model(model, folder)
            restored, _ = load_model(folder)
            for a, b in zip(model.decoder.parameters(), restored.decoder.parameters()):
                torch.testing.assert_close(a.half().float(), b, rtol=0, atol=0)
            for pa, pb in zip(model.pyramids, restored.pyramids):
                for a, b in zip(pa, pb):
                    torch.testing.assert_close(a.texture(), b.texture(), rtol=0, atol=0)
            second = Path(temp) / "saved_again"
            meta2 = export_model(restored, second)
            for a, b in zip(meta["files"], meta2["files"]):
                self.assertEqual(a["sha256"], b["sha256"])
            self.assertTrue(replay_check(restored, restored)["strict_pass"])
            p = folder / "decoder_fp16.bin"
            p.write_bytes(p.read_bytes()[:-2])
            with self.assertRaises(ValueError):
                load_model(folder)


if __name__ == "__main__":
    unittest.main()
