"""Literal diagnostic for Eqs. (7)-(9), arXiv:2311.16121v2.

This is a formula oracle, not a claim about undisclosed author code and not a
replacement for a signed BC6H decoder. In particular Eq. (8)'s first branch
uses the unbarred e1 and Eq. (9) is applied to negative y exactly as printed.
No endpoint/index normalization, saturation, sign correction, or rounding is
inserted here. The caller supplies x as it appears in Eq. (8).
"""

import torch


def paper_eq7(endpoints, *, signed, endpoint_bits=6):
    """Return the paper's barred endpoints in the input tensor's dtype."""
    a = 31.0 / (32.0 if signed else 64.0)
    return (a * (2**16) * endpoints + (2**15)) / (2**endpoint_bits)


def paper_eq8(endpoints, x, first_region, *, signed, endpoint_bits=6, index_bits=3):
    """Return Eq. (8), literally, for [B,4,C] endpoints and [B,N] x/mask."""
    if endpoints.ndim != 3 or endpoints.shape[1] != 4:
        raise ValueError("endpoints must have shape [blocks, 4, channels]")
    if x.ndim != 2 or x.shape != first_region.shape or x.shape[0] != endpoints.shape[0]:
        raise ValueError("x and first_region must have matching [blocks, texels] shapes")
    e = paper_eq7(endpoints, signed=signed, endpoint_bits=endpoint_bits)
    t = (2**index_bits) * x[..., None]
    first = endpoints[:, 0, None, :] + t * (e[:, 1, None, :] - e[:, 0, None, :])
    second = e[:, 2, None, :] + t * (e[:, 3, None, :] - e[:, 2, None, :])
    return torch.where(first_region[..., None].bool(), first, second)


def paper_eq9(y):
    """Return Eq. (9) as printed, including for negative y; no sign extension."""
    h = (torch.floor((y - 1) / 1024) - 1).clamp_min(0)
    return torch.exp2(h - 14) * (y / 1024 - h)


def paper_literal_forward(endpoints, x, first_region, *, signed, endpoint_bits=6, index_bits=3):
    y = paper_eq8(endpoints, x, first_region, signed=signed,
                  endpoint_bits=endpoint_bits, index_bits=index_bits)
    return paper_eq9(y)
