from typing import Callable

import torch
from torch import nn

from bayne.bounds.core import Bounds
from bayne.bounds.ibp import SampleIntervalBoundPropagation


class CROWNIntervalBoundPropagation(Bounds):
    def __init__(self, adaptive_relu=True):
        self.adaptive_relu = adaptive_relu
        self.ibp = SampleIntervalBoundPropagation()

    @torch.no_grad()
    def interval_bounds(self, model, input_bounds):
        lower, upper = input_bounds
        (Omega_0, Omega_accumulator), (Gamma_0, Gamma_accumulator) = self.linear_bounds(model, input_bounds)

        lower, upper = lower.unsqueeze(-2), upper.unsqueeze(-2)

        # We can do this instead of finding the Q-norm, as we only deal with perturbation over a hyperrectangular input,
        # and not a B_p(epsilon) ball
        min_Omega_x = (Omega_0 * torch.where(Omega_0 < 0, upper, lower)).sum(dim=-1)
        max_Gamma_x = (Gamma_0 * torch.where(Gamma_0 < 0, lower, upper)).sum(dim=-1)

        return min_Omega_x + Omega_accumulator, max_Gamma_x + Gamma_accumulator

    @torch.no_grad()
    def linear_bounds(self, model, input_bounds):
        alpha, beta = self.compute_alpha_beta(model, input_bounds)
        linear_bounds = self.compute_linear_bounds(model, alpha, beta)

        return linear_bounds

    def compute_alpha_beta(self, model, input_bounds):
        LBs, UBs = self.ibp.interval_bounds(model, input_bounds, include_intermediate=True)

        alpha_lower, alpha_upper = [], []
        beta_lower, beta_upper = [], []

        for k, module in enumerate(model):
            LB, UB = LBs[k - 1], UBs[k - 1]  # LB, UB from previous linear layer

            negative_regime = UB <= 0
            positive_regime = LB >= 0
            cross_regime = (LB < 0) & (0 < UB)

            if isinstance(module, nn.ReLU):
                alpha_lower_k, alpha_upper_k, beta_lower_k, beta_upper_k = \
                    self.compute_alpha_beta_relu(LB, UB, negative_regime, positive_regime, cross_regime)
            elif isinstance(module, nn.Sigmoid):
                alpha_lower_k, alpha_upper_k, beta_lower_k, beta_upper_k = \
                    self.compute_alpha_beta_sigmoid(LB, UB, negative_regime, positive_regime, cross_regime)
            elif isinstance(module, nn.Tanh):
                alpha_lower_k, alpha_upper_k, beta_lower_k, beta_upper_k = \
                    self.compute_alpha_beta_tanh(LB, UB, negative_regime, positive_regime, cross_regime)
            else:
                continue

            alpha_lower.append(alpha_lower_k)
            alpha_upper.append(alpha_upper_k)
            beta_lower.append(beta_lower_k)
            beta_upper.append(beta_upper_k)

        return (alpha_lower, alpha_upper), (beta_lower, beta_upper)

    def compute_alpha_beta_relu(self, LB, UB, negative_regime, positive_regime, cross_regime):
        alpha_lower_k = torch.zeros_like(LB)
        alpha_upper_k = torch.zeros_like(LB)
        beta_lower_k = torch.zeros_like(LB)
        beta_upper_k = torch.zeros_like(LB)

        alpha_lower_k[negative_regime] = 0
        alpha_upper_k[negative_regime] = 0
        beta_lower_k[negative_regime] = 0
        beta_upper_k[negative_regime] = 0

        alpha_lower_k[positive_regime] = 1
        alpha_upper_k[positive_regime] = 1
        beta_lower_k[positive_regime] = 0
        beta_upper_k[positive_regime] = 0

        LB, UB = LB[cross_regime], UB[cross_regime]

        z = UB / (UB - LB)
        if self.adaptive_relu:
            a = (UB >= torch.abs(LB)).to(torch.float)
        else:
            a = z

        alpha_lower_k[cross_regime] = a
        alpha_upper_k[cross_regime] = z
        beta_lower_k[cross_regime] = 0
        beta_upper_k[cross_regime] = -LB * z

        return alpha_lower_k, alpha_upper_k, beta_lower_k, beta_upper_k

    def compute_alpha_beta_sigmoid(self, LB, UB, n, p, np):
        return self.compute_alpha_beta_general(LB, UB, n, p, np, torch.sigmoid)

    def compute_alpha_beta_tanh(self, LB, UB, n, p, np):
        return self.compute_alpha_beta_general(LB, UB, n, p, np, torch.tanh)

    def compute_alpha_beta_general(self, LB, UB, n, p, np, func):
        @torch.enable_grad()
        def derivative(d):
            d.requires_grad_()
            y = func(d)
            y.backward(torch.ones_like(d))

            return d.grad

        alpha_lower_k = torch.zeros_like(LB)
        alpha_upper_k = torch.zeros_like(LB)
        beta_lower_k = torch.zeros_like(LB)
        beta_upper_k = torch.zeros_like(LB)

        LB_act, UB_act = func(LB), func(UB)

        d = (LB + UB) * 0.5  # Let d be the midpoint of the two bounds
        d_act = func(d)
        d_prime = derivative(d)

        concave_slope = torch.nan_to_num((UB_act - LB_act) / (UB - LB), nan=0.0)

        # Negative regime
        alpha_lower_k[n] = d_prime[n]
        alpha_upper_k[n] = concave_slope[n]
        beta_lower_k[n] = d_act[n] - alpha_lower_k[n] * d[n]
        beta_upper_k[n] = UB_act[n] - alpha_upper_k[n] * UB[n]

        # Positive regime
        alpha_lower_k[p] = concave_slope[p]
        alpha_upper_k[p] = d_prime[p]
        beta_lower_k[p] = LB_act[p] - alpha_lower_k[p] * LB[p]
        beta_upper_k[p] = d_act[p] - alpha_upper_k[p] * d[p]

        # Crossing zero
        LB_np, UB_np = LB[np], UB[np]

        def f_lower(d):
            return derivative(d) - (func(UB_np) - func(d)) / (UB_np - d)

        def f_upper(d):
            return (func(d) - func(LB_np)) / (d - LB_np) - derivative(d)

        d_lower = self.bisection(LB_np, torch.zeros_like(LB_np), f_lower)
        d_upper = self.bisection(torch.zeros_like(UB_np), UB_np, f_upper)

        alpha_lower_k[np] = derivative(d_lower)
        alpha_upper_k[np] = derivative(d_upper)
        beta_lower_k[np] = UB_act[np] - alpha_lower_k[np] * UB_np
        beta_upper_k[np] = LB_act[np] - alpha_upper_k[np] * LB_np

        return alpha_lower_k, alpha_upper_k, beta_lower_k, beta_upper_k

    def bisection(self, l: torch.Tensor, h: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor], num_iter=10) -> torch.Tensor:
        midpoint = (l + h) / 2
        y = f(midpoint)

        for _ in range(num_iter):
            l[y <= 0] = midpoint[y <= 0]
            h[y > 0] = midpoint[y > 0]

            midpoint = (l + h) / 2
            y = f(midpoint)

        return midpoint

    def compute_linear_bounds(self, model, alpha, beta):
        output_size = model[-1].weight.size(0)
        linear_modules = [module for module in model if isinstance(module, nn.Linear)]
        num_linear = len(linear_modules)
        device = linear_modules[0].weight.device

        Omega_k = torch.eye(output_size, device=device).unsqueeze(0)
        Omega_weight = None
        Omega_accumulator = 0

        Gamma_k = torch.eye(output_size, device=device).unsqueeze(0)
        Gamma_weight = None
        Gamma_accumulator = 0

        for k, module in reversed(list(zip(range(1, num_linear + 1), linear_modules))):
            if isinstance(module, nn.Linear):
                bias_k = module.bias if module.bias else torch.zeros((1,), device=device)
                weight_k = module.weight.unsqueeze(0)

                # Lower bound
                # theta_k = self._theta(k, num_linear, Omega_weight, beta, device)
                # bias_theta_k = bias_k + theta_k
                # Omega_accumulator = Omega_accumulator + (Omega_k * bias_theta_k).sum(dim=-1)
                Omega_accumulator = Omega_accumulator + torch.matmul(Omega_k, bias_k)
                if k < num_linear:
                    theta_k = self._theta(k, num_linear, Omega_weight, beta, device)
                    Omega_accumulator += (Omega_weight * theta_k).sum(dim=-1)

                Omega_weight = torch.matmul(Omega_k, weight_k)
                omega_k = self._omega(k - 1, Omega_weight, alpha, device)
                Omega_k = Omega_weight * omega_k

                # Upper bound
                # delta_k = self._delta(k, num_linear, Gamma_weight, beta, device)
                # bias_delta_k = bias_k + delta_k
                # Gamma_accumulator = Gamma_accumulator + (Gamma_k * bias_delta_k).sum(dim=-1)
                Gamma_accumulator = Gamma_accumulator + torch.matmul(Gamma_k, bias_k)
                if k < num_linear:
                    delta_k = self._delta(k, num_linear, Gamma_weight, beta, device)
                    Gamma_accumulator += (Gamma_weight * delta_k).sum(dim=-1)

                Gamma_weight = torch.matmul(Gamma_k, weight_k)
                lambda_k = self._lambda(k - 1, Gamma_weight, alpha, device)
                Gamma_k = Gamma_weight * lambda_k

        return (Omega_k, Omega_accumulator), (Gamma_k, Gamma_accumulator)

    def _delta(self, k, m, gamma_weight, beta, device):
        if k == m:
            # No beta for the last layer
            return torch.tensor([0], device=device)
        else:
            # Zero-indexing
            return torch.where(gamma_weight < 0, beta[0][k - 1].unsqueeze(-2), beta[1][k - 1].unsqueeze(-2))

    def _lambda(self, k, gamma_weight, alpha, device):
        if k == 0:
            return torch.tensor([1], device=device)
        else:
            # Zero-indexing
            return torch.where(gamma_weight < 0, alpha[0][k - 1].unsqueeze(-2), alpha[1][k - 1].unsqueeze(-2))

    def _theta(self, k, m, omega_weight, beta, device):
        if k == m:
            # No beta for the last layer
            return torch.tensor([0], device=device)
        else:
            # Zero-indexing
            return torch.where(omega_weight < 0, beta[1][k - 1].unsqueeze(-2), beta[0][k - 1].unsqueeze(-2))

    def _omega(self, k, omega_weight, alpha, device):
        if k == 0:
            return torch.tensor([1], device=device)
        else:
            # No alpha for the last layer + zero-indexing
            return torch.where(omega_weight < 0, alpha[1][k - 1].unsqueeze(-2), alpha[0][k - 1].unsqueeze(-2))
