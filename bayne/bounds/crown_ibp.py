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
        linear_bounds = self.linear_bounds(model, input_bounds)
        # TODO: Solve LP to find which corner is optimal

    @torch.no_grad()
    def linear_bounds(self, model, input_bounds):
        alpha, beta = self.compute_alpha_beta(model, input_bounds)
        linear_bounds = self.compute_linear_bounds(model, alpha, beta)

        return linear_bounds

    def compute_alpha_beta(self, model, input_bounds):
        LBs, UBs = self.ibp.interval_bounds(model, input_bounds)

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

        if self.adaptive_relu:
            LB, UB = LB[cross_regime], UB[cross_regime]
            a = UB / (UB - LB)
        else:
            a = (UB[cross_regime] >= torch.abs(LB[cross_regime])).to(torch.float)

        alpha_lower_k[cross_regime] = a
        alpha_upper_k[cross_regime] = a
        beta_lower_k[cross_regime] = 0
        beta_upper_k[cross_regime] = -LB[cross_regime]

        return alpha_lower_k, alpha_upper_k, beta_lower_k, beta_upper_k

    def compute_alpha_beta_sigmoid(self, LB, UB, n, p, np):
        def derivative(x):
            s = torch.sigmoid(x)
            return s - s ** 2

        def double_derivative(x):
            s = torch.sigmoid(x)
            return s - 3 * s ** 2 + s ** 3

        return self.compute_alpha_beta_general(LB, UB, n, p, np, torch.sigmoid, derivative, double_derivative)

    def compute_alpha_beta_tanh(self, LB, UB, n, p, np):
        def derivative(x):
            s = torch.tanh(x)
            return 1 - s ** 2

        def double_derivative(x):
            s = torch.tanh(x)
            return -2 * s + 2 * s ** 3

        return self.compute_alpha_beta_general(LB, UB, n, p, np, torch.tanh, derivative, double_derivative)

    def compute_alpha_beta_general(self, LB, UB, n, p, np, func, derivative, double_derivative):
        alpha_lower_k = torch.zeros_like(LB)
        alpha_upper_k = torch.zeros_like(LB)
        beta_lower_k = torch.zeros_like(LB)
        beta_upper_k = torch.zeros_like(LB)

        LB_act, UB_act = func(LB), func(UB)

        d = (LB + UB) * 0.5  # Let d be the midpoint of the two bounds
        d_act = func(d)
        d_prime = derivative(d)

        concave_slope = (LB_act - UB_act) / (UB - LB)

        # Negative regime
        alpha_lower_k[n] = d_prime[n]
        alpha_upper_k[n] = concave_slope[n]
        beta_lower_k[n] = (d_act[n] - alpha_lower_k[n] * d[n]) / alpha_lower_k[n]
        beta_upper_k[n] = LB_act[n] / alpha_upper_k[n] - LB[n]

        # Positive regime
        alpha_lower_k[p] = concave_slope[p]
        alpha_upper_k[p] = d_prime[p]
        beta_lower_k[p] = LB_act[p] / alpha_lower_k[p] - LB[p]
        beta_upper_k[p] = (d_act[p] - alpha_upper_k[p] * d[p]) / alpha_upper_k[p]

        # Crossing zero
        def f_lower(d):
            return (func(UB) - func(d)) / (UB - d) - derivative(d)

        def df_lower(d):
            return -double_derivative(d) - derivative(d) / (UB - d) + (func(UB) - func(d)) / (UB - d) ** 2

        def f_upper(d):
            return (func(d) - func(LB)) / (d - LB) - derivative(d)

        def df_upper(d):
            return -double_derivative(d) + derivative(d) / (d - LB) - (func(d) - func(LB)) / (d - LB) ** 2

        d_lower = self.newton_raphson(torch.zeros_like(LB[np]), f_lower, df_lower)
        d_upper = self.newton_raphson(torch.zeros_like(LB[np]), f_upper, df_upper)

        alpha_lower_k[np] = derivative(d_lower)
        alpha_upper_k[np] = derivative(d_upper)
        beta_lower_k[np] = UB_act[np] / alpha_lower_k[np] - UB[np]
        beta_upper_k[np] = LB_act[np] / alpha_upper_k[np] - LB[np]

        return alpha_lower_k, alpha_upper_k, beta_lower_k, beta_upper_k

    def newton_raphson(self, x, f, df, num_iter=10):
        for _ in range(num_iter):
            x -= f(x) / df(x)

        return x

    def compute_linear_bounds(self, model, alpha, beta):
        output_size = model[-1].weight.size(0)
        linear_modules = [module for module in model if isinstance(module, nn.Linear)]
        num_linear = len(linear_modules)

        Gamma_k = torch.eye(output_size)
        Gamma_weight = None
        Gamma_accumulator = 0

        Omega_k = torch.eye(output_size)
        Omega_weight = None
        Omega_accumulator = 0

        for k, module in reversed(list(zip(range(1, num_linear + 1), linear_modules))):
            if isinstance(module, nn.Linear):
                # Upper bound
                bias_delta_k = (module.bias + self._delta(k, num_linear, Gamma_weight, None if k == num_linear else beta[k])).permute(-1, -2)
                Gamma_accumulator += (Gamma_k * bias_delta_k).sum(dim=-1)

                Gamma_weight = torch.matmul(Gamma_k, module.weight)
                lambda_k = self._lambda(k, Gamma_weight, alpha[k - 1])
                Gamma_k = Gamma_weight * lambda_k

                # Lower bound
                bias_theta_k = (module.bias + self._theta(k, num_linear, Omega_weight, None if k == num_linear else beta[k])).permute(-1, -2)
                Omega_accumulator += (Omega_k * bias_theta_k).sum(dim=-1)

                Omega_weight = torch.matmul(Omega_k, module.weight)
                omega_k = self._omega(k, Omega_weight, alpha[k - 1])
                Omega_k = Omega_weight * omega_k

        return (Omega_k, Omega_accumulator), (Gamma_k, Gamma_accumulator)

    def _delta(self, k, m, gamma_weight, beta):
        if k == m:
            return 0
        else:
            return torch.where(gamma_weight < 0, beta[0], beta[1])

    def _lambda(self, k, gamma_weight, alpha):
        if k == 0:
            return 1
        else:
            return torch.where(gamma_weight < 0, alpha[0], alpha[1])

    def _theta(self, k, m, omega_weight, beta):
        if k == m:
            return 0
        else:
            return torch.where(omega_weight < 0, beta[1], beta[0])

    def _omega(self, k, omega_weight, alpha):
        if k == 0:
            return 1
        else:
            return torch.where(omega_weight < 0, alpha[1], alpha[0])
