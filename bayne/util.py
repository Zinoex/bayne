import functools
import random
import time
from typing import List

import numpy as np
import torch
from torch import Tensor


@torch.jit.script
def tensorlist_neg(input: List[Tensor]):
    return [-x for x in input]


@torch.jit.script
def tensorlist_inplace_add_list(input: List[Tensor], other: List[Tensor]):
    for x, x_prime in zip(input, other):
        x.add_(x_prime)


@torch.jit.script
def tensorlist_inplace_add_float(input: List[Tensor], other: float):
    for x in input:
        x.add_(other)


@torch.jit.script
def tensorlist_inplace_sub_list(input: List[Tensor], other: List[Tensor]):
    for x, x_prime in zip(input, other):
        x.sub_(x_prime)


@torch.jit.script
def tensorlist_inplace_sub_float(input: List[Tensor], other: float):
    for x in input:
        x.sub_(other)


@torch.jit.script
def tensorlist_inplace_truediv_list(input: List[Tensor], other: List[Tensor]):
    for x, x_prime in zip(input, other):
        x.div_(x_prime)


@torch.jit.script
def tensorlist_inplace_truediv_float(input: List[Tensor], other: float):
    for x in input:
        x.div_(other)


@torch.jit.script
def tensorlist_inplace_mul_list(input: List[Tensor], other: List[Tensor]):
    for x, x_prime in zip(input, other):
        x.mul_(x_prime)

@torch.jit.script
def tensorlist_inplace_mul_float(input: List[Tensor], other: float):
    for x in input:
        x.mul_(other)


@torch.jit.script
def tensorlist_truediv(input: List[Tensor], other: float):
    return [x / other for x in input]


@torch.jit.script
def tensorlist_mul(input: List[Tensor], other: float):
    return [x * other for x in input]


@torch.jit.script
def tensorlist_inplace_normal(input: List[Tensor], mu: float, sigma: float):
    for x in input:
        x.normal_(mu, sigma)


@torch.jit.script
def tensorlist_zeroes_like(input: List[Tensor]):
    return [torch.zeros_like(x) for x in input]


class TensorList(List[Tensor]):
    def __neg__(self):
        return TensorList(tensorlist_neg(self))

    def __iadd__(self, other):
        if isinstance(other, TensorList):
            tensorlist_inplace_add_list(self, other)
        else:
            tensorlist_inplace_add_float(self, other)
        return self

    def __isub__(self, other):
        if isinstance(other, TensorList):
            tensorlist_inplace_sub_list(self, other)
        else:
            tensorlist_inplace_sub_float(self, other)
        return self

    def __itruediv__(self, other):
        if isinstance(other, TensorList):
            tensorlist_inplace_truediv_list(self, other)
        else:
            tensorlist_inplace_truediv_float(self, other)
        return self

    def __imul__(self, other):
        if isinstance(other, TensorList):
            tensorlist_inplace_mul_list(self, other)
        else:
            tensorlist_inplace_mul_float(self, other)
        return self

    def __truediv__(self, other):
        return TensorList(tensorlist_truediv(self, other))

    def __mul__(self, other):
        return TensorList(tensorlist_mul(self, other))

    def normal_(self, mu, sigma):
        tensorlist_inplace_normal(self, mu, sigma)

    @staticmethod
    def zeroes_like(x):
        return TensorList(tensorlist_zeroes_like(x))

class TensorList(List[Tensor]):
    def __neg__(self):
        return tensorlist_neg(self)

    def __iadd__(self, other):
        tensorlist_inplace_add(self, other)
        return self

    def __isub__(self, other):
        tensorlist_inplace_sub(self, other)
        return self

    def __itruediv__(self, other):
        tensorlist_inplace_truediv(self, other)
        return self

    def __imul__(self, other):
        tensorlist_inplace_mul(self, other)
        return self

    def __truediv__(self, other):
        return tensorlist_truediv(self, other)

    def __mul__(self, other):
        return tensorlist_mul(self, other)

    def normal_(self, mu, sigma):
        tensorlist_inplace_normal(self, mu, sigma)

    @staticmethod
    def zeroes_like(x):
        tensorlist_zeroes_like(x)


class ResetableModule:
    def reset(self):
        def _reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        self.apply(_reset)


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
