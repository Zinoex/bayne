import functools
import random
import time

import numpy as np
import torch


class TensorList(list):
    def __neg__(self):
        return TensorList([-x for x in self])

    def __iadd__(self, other):
        if isinstance(other, TensorList):
            for x, x_prime in zip(self, other):
                x.add_(x_prime)
        else:
            for x in self:
                x.add_(other)

        return self

    def __isub__(self, other):
        if isinstance(other, TensorList):
            for x, x_prime in zip(self, other):
                x.sub_(x_prime)
        else:
            for x in self:
                x.sub_(other)

        return self

    def __itruediv__(self, other):
        if isinstance(other, TensorList):
            for x, x_prime in zip(self, other):
                x.div_(x_prime)
        else:
            for x in self:
                x.div_(other)

    def __imul__(self, other):
        if isinstance(other, TensorList):
            for x, x_prime in zip(self, other):
                x.mul_(x_prime)
        else:
            for x in self:
                x.mul_(other)

    def __truediv__(self, other):
        return TensorList([x / other for x in self])

    def __mul__(self, other):
        return TensorList([x * other for x in self])


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
