"""Compare simplified region selection with the prior indexed formulation."""
import unittest
import torch
from bcf6.paper_forward import paper_signed_endpoints, paper_signed_halfcode, paper_signed_block
from bcf6.model import configure_block_backend


def old_indexed_block(endpoints, indices, subset):
    e = paper_signed_endpoints(endpoints)
    row = torch.arange(len(e), device=e.device)[:, None]
    a, b = e[row, 2 * subset], e[row, 2 * subset + 1]
    return paper_signed_halfcode(a + indices.clamp(0.0, 1.0)[..., None] * (b-a))


class SimplifiedBlockTests(unittest.TestCase):
    def test_values_and_gradients_match_previous_selection(self):
        torch.manual_seed(11)
        e = (torch.rand(300, 4, 3, dtype=torch.float64) * 4 - 2).requires_grad_()
        t = torch.rand(300, 16, dtype=torch.float64, requires_grad=True)
        subset = torch.randint(0, 2, (300, 16))
        weight = torch.randn(300, 16, 3, dtype=torch.float64)
        old = old_indexed_block(e, t, subset)
        old_grad = torch.autograd.grad((old * weight).sum(), (e, t))
        new = paper_signed_block(e, t, subset)
        new_grad = torch.autograd.grad((new * weight).sum(), (e, t))
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        for expected, actual in zip(old_grad, new_grad):
            torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_backend_selection_rejects_unknown(self):
        configure_block_backend('eager')
        with self.assertRaises(ValueError):
            configure_block_backend('unknown')

if __name__ == '__main__':
    unittest.main()
