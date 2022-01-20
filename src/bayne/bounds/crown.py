from typing import Callable, Tuple

import torch
from torch import nn

from bayne.bounds.util import notnull, add_method


def crown(model):
    @torch.no_grad()
    def crown(self: nn.Sequential, lower, upper):
        LBs, UBs = [lower], [upper]

        modules = surrogate_model(self)

        for i in range(1, len(modules)):
            subnetwork = modules[:i]
            lb, ub = subnetwork_crown(subnetwork, LBs, UBs, lower.device)
            LBs.append(lb)
            UBs.append(ub)

            batch_size = LBs[0].size(0)

        alpha, beta = alpha_beta_sequential(modules, LBs, UBs)
        bounds = linear_bounds(modules, alpha, beta, batch_size, lower.device)

        return bounds

    add_method(model, 'crown', crown)

    return model


def surrogate_model(model):
    with notnull(getattr(model, '_pyro_context', None)):
        layers = []

        for module in model:
            with notnull(getattr(module, '_pyro_context', None)):
                if isinstance(module, nn.Linear):
                    layers.append(SurrogateLinear(module.weight, module.bias))
                else:
                    layers.append(module)

        return layers


class SurrogateLinear:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias
        self.out_features = weight.size(-2)


def subnetwork_crown(model, LBs, UBs, device):
    batch_size = LBs[0].size(0)

    alpha, beta = alpha_beta_sequential(model, LBs, UBs)
    bounds = linear_bounds(model, alpha, beta, batch_size, device)

    return interval_bounds(bounds, LBs[0], UBs[0])


def interval_bounds(bounds, lower, upper):
    (Omega_0, Omega_accumulator), (Gamma_0, Gamma_accumulator) = bounds

    lower, upper = lower.unsqueeze(-2), upper.unsqueeze(-2)

    # We can do this instead of finding the Q-norm, as we only deal with perturbation over a hyperrectangular input,
    # and not a B_p(epsilon) ball
    mid = (lower + upper) / 2
    diff = (upper - lower) / 2

    min_Omega_x = (torch.matmul(Omega_0, mid) - torch.matmul(torch.abs(Omega_0), diff))[..., 0]
    max_Gamma_x = (torch.matmul(Gamma_0, mid) + torch.matmul(torch.abs(Gamma_0), diff))[..., 0]

    return min_Omega_x + Omega_accumulator, max_Gamma_x + Gamma_accumulator


def alpha_beta_sequential(model, LBs, UBs):
    alpha, beta = [], []

    for module, LB, UB in zip(model, LBs, UBs):
        assert torch.all(LB <= UB + 1e-6)

        n = UB <= 0
        p = 0 <= LB
        np = (LB < 0) & (0 < UB)

        if isinstance(module, nn.ReLU):
            a, b = alpha_beta_relu(LB, UB, n, p, np)
        elif isinstance(module, nn.Sigmoid):
            a, b = alpha_beta_sigmoid(LB, UB, n, p, np)
        elif isinstance(module, nn.Tanh):
            a, b = alpha_beta_tanh(LB, UB, n, p, np)
        else:
            a, b = (None, None), (None, None)

        alpha.append(a)
        beta.append(b)

    return alpha, beta


def alpha_beta_relu(LB, UB, n, p, np, adaptive_relu=True):
    alpha_lower_k = torch.zeros_like(LB)
    alpha_upper_k = torch.zeros_like(LB)
    beta_lower_k = torch.zeros_like(LB)
    beta_upper_k = torch.zeros_like(LB)

    alpha_lower_k[n] = 0
    alpha_upper_k[n] = 0
    beta_lower_k[n] = 0
    beta_upper_k[n] = 0

    alpha_lower_k[p] = 1
    alpha_upper_k[p] = 1
    beta_lower_k[p] = 0
    beta_upper_k[p] = 0

    LB, UB = LB[np], UB[np]

    z = UB / (UB - LB)
    if adaptive_relu:
        a = (UB >= torch.abs(LB)).to(torch.float)
    else:
        a = z

    alpha_lower_k[np] = a
    alpha_upper_k[np] = z
    beta_lower_k[np] = 0
    beta_upper_k[np] = -LB * z

    return (alpha_lower_k, alpha_upper_k), (beta_lower_k, beta_upper_k)


def alpha_beta_sigmoid(LB, UB, n, p, np):
    def derivative(d):
        return torch.sigmoid(d) * (1 - torch.sigmoid(d))

    return alpha_beta_general(LB, UB, n, p, np, torch.sigmoid, derivative)


def alpha_beta_tanh(LB, UB, n, p, np):
    def derivative(d):
        return 1 - torch.tanh(d) ** 2

    return alpha_beta_general(LB, UB, n, p, np, torch.tanh, derivative)


def alpha_beta_general(LB, UB, n, p, np, func, derivative):
    alpha_lower_k = torch.zeros_like(LB)
    alpha_upper_k = torch.zeros_like(LB)
    beta_lower_k = torch.zeros_like(LB)
    beta_upper_k = torch.zeros_like(LB)

    LB_act, UB_act = func(LB), func(UB)
    LB_prime, UB_prime = derivative(LB), derivative(UB)

    d = (LB + UB) * 0.5  # Let d be the midpoint of the two bounds
    d_act = func(d)
    d_prime = derivative(d)

    slope = (UB_act - LB_act) / (UB - LB)

    # Negative regime
    alpha_lower_k[n] = d_prime[n]
    alpha_upper_k[n] = slope[n]
    beta_lower_k[n] = d_act[n] - alpha_lower_k[n] * d[n]
    beta_upper_k[n] = UB_act[n] - alpha_upper_k[n] * UB[n]

    # Positive regime
    alpha_lower_k[p] = slope[p]
    alpha_upper_k[p] = d_prime[p]
    beta_lower_k[p] = LB_act[p] - alpha_lower_k[p] * LB[p]
    beta_upper_k[p] = d_act[p] - alpha_upper_k[p] * d[p]

    #################
    # Crossing zero #
    #################
    # Upper
    UB_prime_at_LB = UB_prime * (LB - UB) + UB_act
    direct_upper = np & (UB_prime_at_LB <= 0)
    implicit_upper = np & (UB_prime_at_LB > 0)

    alpha_upper_k[direct_upper] = slope[direct_upper]

    def f_upper(d):
        a_slope = (func(d) - func(LB[implicit_upper])) / (d - LB[implicit_upper])
        a_derivative = derivative(d)
        return a_slope - a_derivative

    d_upper = bisection(torch.zeros_like(UB[implicit_upper]), UB[implicit_upper], f_upper)

    alpha_upper_k[implicit_upper] = derivative(d_upper[0])
    beta_upper_k[np] = LB_act[np] - alpha_upper_k[np] * LB[np]

    # Lower
    LB_prime_at_UB = LB_prime * (UB - LB) + LB_act
    direct_lower = np & (LB_prime_at_UB >= 0)
    implicit_lower = np & (LB_prime_at_UB < 0)

    alpha_lower_k[direct_lower] = slope[direct_lower]

    def f_lower(d):
        a_slope = (func(UB[implicit_lower]) - func(d)) / (UB[implicit_lower] - d)
        a_derivative = derivative(d)
        return a_derivative - a_slope

    d_lower = bisection(LB[implicit_lower], torch.zeros_like(LB[implicit_lower]), f_lower)

    alpha_lower_k[implicit_lower] = derivative(d_lower[1])
    beta_lower_k[np] = UB_act[np] - alpha_lower_k[np] * UB[np]

    return (alpha_lower_k, alpha_upper_k), (beta_lower_k, beta_upper_k)


def bisection(l: torch.Tensor, h: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], num_iter=20) -> Tuple[torch.Tensor, torch.Tensor]:
    midpoint = (l + h) / 2

    for _ in range(num_iter):
        y = f(midpoint)

        l[y <= 0] = midpoint[y <= 0]
        h[y > 0] = midpoint[y > 0]

        midpoint = (l + h) / 2

    return l, h


def linear_bounds(model, alpha, beta, batch_size, device):
    model = surrogate_model(model)

    # Compute bounds as two iterations to reduce memory consumption by half
    return oneside_linear_bound(model, alpha, beta, batch_size, device, act_lower), \
           oneside_linear_bound(model, alpha, beta, batch_size, device, act_upper)


def oneside_linear_bound(model, alpha, beta, batch_size, device, act_fn):
    out_size = output_size(model)

    W_tilde = torch.eye(out_size, device=device).unsqueeze(0).expand(batch_size, out_size, out_size)
    acc = 0

    # List is necessary around zip to allow reversing
    for module, (al_k, au_k), (bl_k, bu_k) in reversed(list(zip(model, alpha, beta))):
        with notnull(getattr(module, '_pyro_context', None)):
            if isinstance(module, (SurrogateLinear, nn.Linear)):
                W_tilde, bias = linear(W_tilde, module)
                acc = acc + bias
            elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
                W_tilde, bias = act_fn(W_tilde, al_k, au_k, bl_k, bu_k)
                acc = acc + bias
            else:
                raise NotImplemented()

    return W_tilde, acc


def output_size(model):
    for module in reversed(model):
        if isinstance(module, (SurrogateLinear, nn.Linear)):
            return module.out_features


def linear(W_tilde, module):
    bias = module.bias
    weight = module.weight

    if weight.dim() == 2:
        W_tilde_new = torch.matmul(W_tilde, weight)
    else:
        if W_tilde.dim() == 3:
            W_tilde = W_tilde.unsqueeze(0)

        W_tilde_new = torch.matmul(W_tilde, weight.unsqueeze(1))

    if bias is None:
        bias_acc = 0
    elif bias.dim() == 1:
        bias_acc = torch.matmul(W_tilde, bias)
    else:
        bias = bias.view(bias.size(0), 1, bias.size(-1), 1)
        bias_acc = torch.matmul(W_tilde, bias)[..., 0]

    return W_tilde_new, bias_acc


def act_lower(Omega_tilde, al_k, au_k, bl_k, bu_k):
    bias = torch.sum(Omega_tilde * _theta(Omega_tilde, bl_k, bu_k), dim=-1)
    Omega_tilde = Omega_tilde * _omega(Omega_tilde, al_k, au_k)

    return Omega_tilde, bias


def _theta(omega_weight, beta_lower, beta_upper):
    return torch.where(omega_weight < 0, beta_upper.unsqueeze(-2), beta_lower.unsqueeze(-2))


def _omega(omega_weight, alpha_lower, alpha_upper):
    return torch.where(omega_weight < 0, alpha_upper.unsqueeze(-2), alpha_lower.unsqueeze(-2))


def act_upper(Gamma_tilde, al_k, au_k, bl_k, bu_k):
    bias = torch.sum(Gamma_tilde * _delta(Gamma_tilde, bl_k, bu_k), dim=-1)
    Gamma_tilde = Gamma_tilde * _lambda(Gamma_tilde, al_k, au_k)

    return Gamma_tilde, bias


def _delta(gamma_weight, beta_lower, beta_upper):
    return torch.where(gamma_weight < 0, beta_lower.unsqueeze(-2), beta_upper.unsqueeze(-2))


def _lambda(gamma_weight, alpha_lower, alpha_upper):
    return torch.where(gamma_weight < 0, alpha_lower.unsqueeze(-2), alpha_upper.unsqueeze(-2))
