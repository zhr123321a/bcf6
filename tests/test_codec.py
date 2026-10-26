import tempfile
import unittest
from pathlib import Path
import numpy as np
from bcf6 import codec


class CodecTests(unittest.TestCase):
    def test_all_partitions_and_signs_against_independent_decoder(self):
        rng = np.random.default_rng(20261026)
        for signed in (False, True):
            n = 4096
            q = rng.integers(
                -32 if signed else 0, 32 if signed else 64, (n, 4, 3), dtype=np.int32
            )
            q[:64] = np.arange(-32 if signed else 0, 32 if signed else 64)[
                :, None, None
            ]
            idx = rng.integers(0, 8, (n, 16), dtype=np.int32)
            p = np.arange(n, dtype=np.int32) % 32
            expected = codec.decode_parameters(q, idx, p, signed)
            payload = codec.pack_blocks(q, idx, p, signed)
            self.assertEqual(len(payload), n * 16)
            q2, i2, p2 = codec.unpack_blocks(payload, signed)
            actual = codec.decode_parameters(q2, i2, p2, signed)
            np.testing.assert_array_equal(actual, expected)
            image = codec.independent_decode(payload, 4 * n, 4, signed)
            np.testing.assert_array_equal(
                image, codec.blocks_to_image(expected, 4 * n, 4)
            )

    def test_dds_mips_and_truncation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "feature.dds"
            mips = []
            for n in [64, 16, 4, 1]:
                mips.append(
                    codec.pack_blocks(
                        np.ones((n, 4, 3), np.int32) * 14,
                        np.zeros((n, 16), np.int32),
                        np.zeros(n, np.int32),
                    )
                )
            receipt = codec.save_dds(path, 32, 32, mips)
            signed, loaded = codec.load_dds(path)
            self.assertTrue(signed)
            self.assertEqual([m[2] for m in loaded], mips)
            self.assertEqual(receipt["payload_bytes"], 1360)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                codec.load_dds(path)

    def test_invalid_parameters_rejected(self):
        q = np.zeros((1, 4, 3), np.int32)
        idx = np.zeros((1, 16), np.int32)
        p = np.zeros(1, np.int32)
        with self.assertRaises(ValueError):
            codec.pack_blocks(q, idx + 8, p)
        with self.assertRaises(ValueError):
            codec.pack_blocks(q + 32, idx, p)
        with self.assertRaises(ValueError):
            codec.unpack_blocks(b"\x00" * 16)


if __name__ == "__main__":
    unittest.main()
