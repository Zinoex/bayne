import torch
import torch.nn.functional as F
from torch import nn

from bayne.distributions import PriorWeightDistribution, PosteriorWeightDistribution


class VariationalBayesianLayer(nn.Module):
    pass


class VariationalLinear(VariationalBayesianLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 prior=PriorWeightDistribution(),
                 device=None, dtype=None):
        super().__init__()

        self.prior = prior

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs).normal_(0, 0.1))
        self.weight_rho = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs).normal_(-7, 0.1))
        self.weight_posterior = PosteriorWeightDistribution(self.weight_mu, self.weight_rho)

        self.include_bias = bias
        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features, **factory_kwargs).normal_(0, 0.1))
            self.bias_rho = nn.Parameter(torch.empty(out_features, **factory_kwargs).normal_(-7, 0.1))
            self.bias_posterior = PosteriorWeightDistribution(self.bias_mu, self.bias_rho)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self.kl_loss = 0
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def forward(self, x):
        if self.frozen:
            return F.linear(x, self.weight_mu, self.bias_mu)

        w = self.weight_posterior.sample()

        self.kl_loss = self.weight_posterior.log_posterior(w) - self.prior.log_prior(w)

        if self.include_bias:
            b = self.bias_posterior.sample()

            self.kl_loss += self.bias_posterior.log_posterior(b) - self.prior.log_prior(b)
        else:
            b = None

        return F.linear(x, w, b)


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
            if isinstance(module, VariationalBayesianLayer):
                module.freeze()
            else:
                for submodule in module.children():
                    _freeze(submodule)

        _freeze(self)

    def unfreeze(self):
        def _unfreeze(module):
            if isinstance(module, VariationalBayesianLayer):
                module.unfreeze()
            else:
                for submodule in module.children():
                    _unfreeze(submodule)

        _unfreeze(self)

    def kl_loss(self):
        def _kl_loss(module):
            if isinstance(module, VariationalBayesianLayer):
                return module.kl_loss
            else:
                child_kl = [_kl_loss(submodule) for submodule in module.children()]

                if len(child_kl) > 0:
                    return torch.stack(child_kl, dim=0).sum(dim=0)
                else:
                    return torch.zeros(1)[0]

        return _kl_loss(self)

    def zero_kl(self):
        def _zero_kl(module):
            if isinstance(module, VariationalBayesianLayer):
                module.kl_loss = 0
            else:
                for submodule in module.children():
                    _zero_kl(submodule)

        _zero_kl(self)

    def sample_elbo(self, idx, x, y, criterion, num_samples, num_batches):
        loss = 0
        for _ in range(num_samples):
            self.zero_kl()

            y_pred = self(x)
            loss += criterion(y_pred, y)

            pi = 1 / num_batches
            loss += self.kl_loss() * pi

        return loss / num_samples
