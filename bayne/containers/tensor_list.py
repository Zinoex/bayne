from typing import List

import torch
from torch import Tensor


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
def tensorlist_inplace_mul_list(input: List[Tensor], other: List[Tensor]):
    for x, x_prime in zip(input, other):
        x.mul_(x_prime)

@torch.jit.script
def tensorlist_inplace_mul_float(input: List[Tensor], other: float):
    for x in input:
        x.mul_(other)


@torch.jit.script
def tensorlist_inplace_normal(input: List[Tensor], mu: float, sigma: float):
    for x in input:
        x.normal_(mu, sigma)


@torch.jit.script
def tensorlist_zeroes_like(input: List[Tensor]):
    return [torch.zeros_like(x) for x in input]


@torch.jit.script
def tensorlist_step(param: List[Tensor], grad: List[Tensor], step_size: float):
    for x, x_prime in zip(param, grad):
        x_prime.mul_(step_size)
        x.add_(x_prime)


@torch.jit.script
def tensorlist_sg_hmc_momentum_update(momentum: List[Tensor], grad: List[Tensor], step_size: float, alpha: float, sigma: float):
    for x, x_prime in zip(momentum, grad):
        # v <- alpha * v - step_size * grad(U) + noise
        x.mul_(alpha)

        x_prime.mul_(step_size)
        x.sub_(x_prime)

        noise = torch.randn_like(x)
        noise.mul_(sigma)
        x.add_(noise)


@torch.jit.script
def tensorlist_sg_ld_step(param: List[Tensor], grad: List[Tensor], step_size: float, sigma: float):
    for x, x_prime in zip(param, grad):
        # v <- v - step_size * grad(U) + noise
        x_prime.mul_(step_size)
        x.sub_(x_prime)

        noise = torch.randn_like(x)
        noise.mul_(sigma)
        x.add_(noise)


class TensorList(List[Tensor]):
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

    def __imul__(self, other):
        if isinstance(other, TensorList):
            tensorlist_inplace_mul_list(self, other)
        else:
            tensorlist_inplace_mul_float(self, other)
        return self

    def __neg__(self):
        return TensorList([-x for x in self])

    def normal_(self, mu, sigma):
        tensorlist_inplace_normal(self, mu, sigma)

    @staticmethod
    def zeroes_like(x):
        return TensorList(tensorlist_zeroes_like(x))

    def step(self, grad, step_size: float):
        tensorlist_step(self, grad, step_size)

        return self

    def sg_hmc_momentum_update(self, grad: List[Tensor], step_size: float, alpha: float, sigma: float):
        tensorlist_sg_hmc_momentum_update(self, grad, step_size, alpha, sigma)

        return self

    def sg_ld_step(self, grad: List[Tensor], step_size: float, sigma: float):
        tensorlist_sg_ld_step(self, grad, step_size, sigma)

        return self
