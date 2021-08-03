import numpy as np
import torch
from torch import nn, distributions
from torch.nn import init
import torch.nn.functional as F


class VariationalLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 prior_sigma1=1.5,
                 prior_sigma2=0.1,
                 prior_pi=0.5,
                 bias: bool = True,
                 device=None, dtype=None):
        super().__init__()

        self.prior_sigma1 = prior_sigma1
        self.prior_sigma2 = prior_sigma2
        self.prior_pi1 = prior_pi
        self.prior_pi2 = 1.0 - prior_pi

        self.prior_dist1 = distributions.Normal(0.0, self.prior_sigma1)
        self.prior_dist2 = distributions.Normal(0.0, self.prior_sigma2)

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # Store rho = log(exp(sigma) - 1) to allow numerical stability and
        # free parameterization to positive variance, R -> R+
        self.weight_rho = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.include_bias = bias
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features, **factory_kwargs))
            self.bias_rho = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.reset_parameters()

        self.kl_loss = 0
        self.frozen = False

    def reset_parameters(self) -> None:
        init_sigma = np.sqrt(self.prior_pi1 * self.prior_sigma1 ** 2 + self.prior_pi2 * self.prior_sigma2 ** 2)

        init.normal(self.weight_mu, 0, init_sigma)
        init.zeros_(self.weight_rho)

        if self.include_bias is not None:
            init.normal(self.bias_mu, 0, init_sigma)
            init.zeros_(self.bias_rho)

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def forward(self, x):
        # Apply sigma = log(1 + exp(rho)) to allow free parameterization to positive variance, R -> R+
        weight_sigma = F.softplus(self.weight_rho, beta=1, threshold=20)
        if self.include_bias:
            bias_sigma = F.softplus(self.bias_rho, beta=1, threshold=20)

        if self.frozen:
            weight = self.weight_mu
            bias = self.bias_mu
        else:
            weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
            if self.include_bias:
                bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)
            else:
                bias = None

        self.update_kl_loss(weight, self.weight_mu, weight_sigma)
        if self.include_bias:
            self.update_kl_loss(bias, self.bias_mu, bias_sigma)

        return F.linear(x, weight, bias)

    def update_kl_loss(self, w, mu, sigma):
        variational_dist = distributions.Normal(mu, sigma)

        log_prior_prob = torch.log(self.prior_pi1 * self.prior_dist1.log_prob(w).exp() +
                                   self.prior_pi2 * self.prior_dist2.log_prob(w).exp())

        self.kl_loss = variational_dist.log_prob(w) - log_prior_prob + self.kl_loss


class BaseVariationalBNN(nn.Module):
    def __init__(self):
        super().__init__()

    def predict_dist(self, *args, num_samples=1, dim=0, **kwargs):
        self.unfreeze()

        preds = torch.stack([self(*args, **kwargs) for _ in range(num_samples)], dim=dim)
        return preds

    def predict_mean(self, *args, num_samples=1, dim=0, **kwargs):
        preds = self.predict_dist(*args, num_samples=num_samples, dim=dim, **kwargs)
        return preds.mean(dim=dim)

    def freeze(self):
        def _freeze(module):
            if hasattr(module, 'freeze'):
                module.freeze()
            else:
                for submodule in module.children():
                    _freeze(submodule)

        _freeze(self)

    def unfreeze(self):
        def _unfreeze(module):
            if hasattr(module, 'unfreeze'):
                module.unfreeze()
            else:
                for submodule in module.children():
                    _unfreeze(submodule)

        _unfreeze(self)
