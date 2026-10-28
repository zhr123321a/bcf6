"""Paper-inspired signed BC6H training forward, with explicit BC6H completions.

This implementation is not a recovered author implementation. Eq. (7)'s
positive-domain affine expression is extended to signed endpoints. Zero and
the saturated edge are anchored to signed Mode 10; the first and last code
intervals are linearly connected so that gradients remain defined there.
Eq. (8)'s two regions both use unquantized endpoints and the internal index is
the [0,1] endpoint weight (paper x = weight / 2**q). Eq. (9) is applied to
the magnitude and its sign restored. See docs/PAPER_FORWARD_SIGNED_ZH.md.
"""

import torch


def paper_signed_endpoints(endpoints):
    """Map continuous signed 6-bit codes to signed half-code coordinates.

    Paper Eq. (7), a=31/32 and b=6, is 992*e+512 for e>0.
    Signed symmetry, exact zero, exact saturated extrema, and the continuous
    transitions over [0,1] and [30,31] are explicit reproduction choices.
    """
    e = endpoints.clamp(-31.0, 31.0)
    magnitude = torch.where(e >= 0, e, -e)
    paper = (31.0 / 32.0 * (2**16) * magnitude + 2**15) / 2**6
    zero_transition = magnitude * 1504.0
    edge_transition = 30272.0 + (magnitude - 30.0) * (31743.0 - 30272.0)
    mapped = torch.where(magnitude < 1.0, zero_transition, paper)
    mapped = torch.where(magnitude > 30.0, edge_transition, mapped)
    return torch.where(e >= 0, mapped, -mapped)


def paper_signed_halfcode(y):
    """Apply paper Eq. (9) on magnitudes and restore sign, including at zero."""
    magnitude = torch.where(y >= 0, y, -y)
    h = (torch.floor((magnitude - 1.0) / 1024.0) - 1.0).clamp_min(0.0)
    value = torch.exp2(h - 14.0) * (magnitude / 1024.0 - h)
    return torch.where(y >= 0, value, -value)


def paper_signed_block(endpoints, indices, subset):
    """Differentiable block decode: [B,4,3], [B,16], [B,16] -> [B,16,3]."""
    e = paper_signed_endpoints(endpoints)
    second = subset[..., None] != 0
    a = torch.where(second, e[:, 2, None, :], e[:, 0, None, :])
    b = torch.where(second, e[:, 3, None, :], e[:, 1, None, :])
    weight = indices.clamp(0.0, 1.0)[..., None]
    y = a + weight * (b - a)
    return paper_signed_halfcode(y)
