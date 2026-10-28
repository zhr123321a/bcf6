"""Targeted checks of the literal paper diagnostic; CPU only."""

import math
import unittest
import torch
from bcf6.paper_formula_literal import paper_eq7, paper_eq8, paper_eq9, paper_literal_forward


def scalar_reference(endpoints, x, first, signed):
    a = 31 / (32 if signed else 64)
    e = [(a * 65536 * v + 32768) / 64 for v in endpoints]
    if first:
        y = endpoints[0] + 8 * x * (e[1] - e[0])
    else:
        y = e[2] + 8 * x * (e[3] - e[2])
    h = max(math.floor((y - 1) / 1024) - 1, 0)
    return y, 2 ** (h - 14) * (y / 1024 - h)


class LiteralPaperTests(unittest.TestCase):
    def test_unsigned_equal_endpoint_asymmetry(self):
        endpoints = torch.ones((1, 4, 1), dtype=torch.float64)
        x = torch.zeros((1, 2), dtype=torch.float64)
        mask = torch.tensor([[True, False]])
        self.assertEqual(paper_eq7(endpoints, signed=False)[0, 0, 0].item(), 1008)
        self.assertEqual(paper_eq8(endpoints, x, mask, signed=False).flatten().tolist(), [1, 1008])

    def test_literal_negative_does_not_equal_signed_half(self):
        y = torch.tensor([-15360.0, 15360.0], dtype=torch.float64)
        result = paper_eq9(y)
        self.assertEqual(result[0].item(), -15 * 2**-14)
        self.assertEqual(result[1].item(), 1.0)

    def test_two_regions_against_scalar_equations(self):
        for signed in (False, True):
            values = [-0.2, 0.35, 0.65, -0.1] if signed else [0.2, 0.35, 0.65, 0.1]
            endpoints = torch.tensor(values, dtype=torch.float64).reshape(1, 4, 1)
            x = torch.tensor([[0.01, 0.01]], dtype=torch.float64)
            mask = torch.tensor([[True, False]])
            y = paper_eq8(endpoints, x, mask, signed=signed).flatten()
            out = paper_literal_forward(endpoints, x, mask, signed=signed).flatten()
            for region in (0, 1):
                expected_y, expected_w = scalar_reference(values, .01, region == 0, signed)
                self.assertAlmostEqual(y[region].item(), expected_y, places=12)
                self.assertAlmostEqual(out[region].item(), expected_w, places=12)

    def test_gradients_inside_smooth_regions(self):
        endpoints = torch.tensor([0.2, 0.35, 0.65, 0.1], dtype=torch.float64,
                                 requires_grad=True).reshape(1, 4, 1)
        x = torch.tensor([[0.01, 0.01]], dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[True, False]])
        self.assertTrue(torch.autograd.gradcheck(
            lambda e, t: paper_literal_forward(e, t, mask, signed=False),
            (endpoints, x), eps=1e-6, atol=1e-8, rtol=1e-5))

if __name__ == "__main__":
    unittest.main()
