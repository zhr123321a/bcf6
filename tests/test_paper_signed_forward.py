"""Checks for signed paper-inspired forward and explicit BC6H completions."""

import struct
import unittest
import numpy as np
import torch
from bcf6.model import FeatureMip, halfcode_to_float
from bcf6.paper_forward import paper_signed_endpoints, paper_signed_halfcode, paper_signed_block
from bcf6.codec import decode_parameters


class SignedPaperForwardTests(unittest.TestCase):
    def test_endpoints_at_zero_interior_and_saturation(self):
        values = torch.tensor([-31., -30., -1., -.5, 0., .5, 1., 30., 31.], dtype=torch.float64)
        expected = torch.tensor([-31743., -30272., -1504., -752., 0., 752., 1504., 30272., 31743.], dtype=torch.float64)
        torch.testing.assert_close(paper_signed_endpoints(values), expected, rtol=0, atol=0)

    def test_halfcode_against_known_signed_half_patterns(self):
        codes = [0, 1, 1024, 2048, 15360, 31743]
        for code in codes:
            expected = struct.unpack('<e', struct.pack('<H', code))[0]
            for sign in (-1, 1):
                actual = paper_signed_halfcode(torch.tensor(float(code * sign), dtype=torch.float64)).item()
                self.assertEqual(actual, expected * sign)

    def test_all_finite_signed_halfcode_bit_patterns(self):
        codes = np.arange(-31743, 31744, dtype=np.int32)
        bits = (np.abs(codes) | np.where(codes < 0, 32768, 0)).astype('<u2')
        expected = bits.view('<f2').astype(np.float32)
        actual = paper_signed_halfcode(torch.from_numpy(codes.astype(np.float32))).numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_region_symmetry_and_model_integration(self):
        mip = FeatureMip(4, 4, mode='block')
        with torch.no_grad():
            mip.endpoints.fill_(1.)
            mip.indices.fill_(.4)
            mip.partitions.zero_()
            mip.refresh_partition_table()
        expected = 1504. * 2**-24
        torch.testing.assert_close(mip.texture(), torch.full((4, 4, 3), expected), rtol=0, atol=0)
        with torch.no_grad():
            mip.endpoints.zero_()
        self.assertTrue(torch.count_nonzero(mip.texture()) == 0)

    def test_paper_floor_expression_is_used_for_continuous_codes(self):
        y = torch.tensor([2048.5], dtype=torch.float64)
        paper_value = paper_signed_halfcode(y)
        self.assertEqual(paper_value.item(), 2048.5 * 2**-24)
        self.assertNotEqual(paper_value.item(), halfcode_to_float(y).item())

    def test_continuous_training_and_discrete_decode_are_distinct(self):
        ends = torch.tensor([[[1.], [2.], [0.], [0.]]])
        weight = torch.tensor([[9. / 64.]])
        out = paper_signed_block(ends, weight, torch.tensor([[0]])).item()
        q = np.array([[[1, 1, 1], [2, 2, 2], [0, 0, 0], [0, 0, 0]]], dtype=np.int32)
        indices = np.ones((1, 16), dtype=np.int32)
        discrete = decode_parameters(q, indices, np.zeros(1, dtype=np.int32), True)[0, 0, 0]
        self.assertEqual(out, 1643.5 * 2**-24)
        self.assertEqual(float(discrete), 1627 * 2**-24)

    def test_gradients_for_signed_endpoints_and_weights(self):
        e = torch.tensor([[[-.3, .25, 1.3], [.7, 1.2, -1.7],
                           [.4, -.8, 2.2], [1.5, 1.7, 2.7]]],
                         dtype=torch.float64, requires_grad=True)
        t = torch.tensor([[.14, .27]], dtype=torch.float64, requires_grad=True)
        subsets = torch.tensor([[0, 1]])
        self.assertTrue(torch.autograd.gradcheck(
            lambda ends, weights: paper_signed_block(ends, weights, subsets),
            (e, t), eps=1e-6, atol=1e-7, rtol=1e-4))

    def test_zero_has_nonzero_endpoint_gradient(self):
        x = torch.tensor([0.], dtype=torch.float64, requires_grad=True)
        y = paper_signed_halfcode(paper_signed_endpoints(x))
        y.backward()
        self.assertEqual(x.grad.item(), 1504. * 2**-24)

if __name__ == '__main__':
    unittest.main()
