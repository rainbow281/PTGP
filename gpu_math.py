"""Device-resident scalar reductions; no implicit CPU/NumPy conversion."""
import math
import torch


def percentile_linear(values, percentage):
    """NumPy-compatible linear percentile, evaluated on values.device.

    kthvalue avoids torch.quantile's input-size limit and a full sort. Only
    public shape/percentile metadata is handled by Python.
    """
    if not math.isfinite(percentage) or not 0 <= percentage <= 100:
        raise ValueError("percentage must be in [0, 100]")
    flat = values.detach().reshape(-1)
    if not flat.numel():
        raise ValueError("Empty percentile input")
    position = (flat.numel() - 1) * percentage / 100.0
    lower = math.floor(position)
    fraction = position - lower
    low = flat.kthvalue(lower + 1).values.double()
    high = flat.kthvalue(min(lower + 2, flat.numel())).values.double()
    result = low + fraction * (high - low)
    return torch.where(torch.isnan(flat).any(), result.new_tensor(float("nan")), result)
