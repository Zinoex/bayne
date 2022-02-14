import types
from contextlib import contextmanager
from typing import Optional, Callable, Tuple, List

import torch


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


# For convenience in specifying type hints and for a semantic name (understanding)
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