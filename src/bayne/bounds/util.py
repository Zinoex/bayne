import types
from contextlib import contextmanager
from typing import Optional, Callable, Tuple, List

import torch


# For convenience in specifying type hints and for a semantic name (understanding)
from torch import nn

from bayne.mcmc import PyroMCMCBNN

OptionalTensor = Optional[torch.Tensor]
TensorFunction = Callable[[torch.Tensor], torch.Tensor]

LinearBound = Tuple[torch.Tensor, torch.Tensor]
LinearBounds = Tuple[LinearBound, LinearBound]
IntervalBounds = Tuple[torch.Tensor, torch.Tensor]

LayerBound = Tuple[torch.Tensor, torch.Tensor]
LayerBounds = List[LayerBound]

AlphaBeta = Tuple[Tuple[OptionalTensor, OptionalTensor], Tuple[OptionalTensor, OptionalTensor]]
AlphaBetas = List[AlphaBeta]

WeightBias = Tuple[torch.Tensor, torch.Tensor]


@contextmanager
def optional(condition, context_manager):
    if condition:
        with context_manager:
            yield
    else:
        yield


@contextmanager
def notnull(context_manager):
    if context_manager is not None:
        with context_manager:
            yield
    else:
        yield


def add_method(class_or_obj, name, func):
    if not isinstance(class_or_obj, type):
        func = types.MethodType(func, class_or_obj)

    setattr(class_or_obj, name, func)


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def validate_affine(model: PyroMCMCBNN, lower: torch.Tensor, upper: torch.Tensor, linear: LinearBounds, num_points=10):
    return

    # TODO: generalize validation to n-D

    points = tensor_linspace(lower, upper, num_points).view(lower.size(0), num_points, -1)

    flat = points.view(-1, 1)
    output = model(flat)

    if output.dim() == 2:
        output = output.view(lower.size(0), num_points, -1)
    else:
        output = output.view(output.size(0), lower.size(0), num_points, -1)

    lower = linear[0][0].unsqueeze(-2).matmul(points.unsqueeze(-1))[..., 0] + linear[0][1].unsqueeze(-1)
    upper = linear[1][0].unsqueeze(-2).matmul(points.unsqueeze(-1))[..., 0] + linear[1][1].unsqueeze(-1)

    assert torch.all(lower <= output + 1e-6)
    assert torch.all(upper >= output - 1e-6)
