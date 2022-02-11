import logging

import torch
from torch import nn, distributions
import torch.nn.functional as F

from pyro.nn import PyroModule, PyroSample
from pyro.infer.mcmc import HMC, MCMC, NUTS
from pyro.infer import Predictive
import pyro.distributions as dist
import pyro


logger = logging.getLogger(__file__)


class PyroBatchLinear(nn.Linear, PyroModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None, weight_prior=None, bias_prior=None) -> None:
        # While calling super().__init__() creates the weights and we overwrite them later, it's just easier this way,
        # and we get to inherit from nn.Linear. It shouldn't be too much of an issue considering that you usually
        # only instantiate a model once during execution.
        super(PyroBatchLinear, self).__init__(in_features, out_features, bias, device, dtype)

        if weight_prior is None:
            weight_prior = dist.Normal(torch.as_tensor(0.0, device=device), torch.as_tensor(1.0, device=device)) \
                                            .expand(self.weight.shape) \
                                            .to_event(self.weight.dim())

        if bias and bias_prior is None:
            bias_prior = dist.Normal(torch.as_tensor(0.0, device=device), torch.as_tensor(0.5, device=device)) \
                                            .expand(self.bias.shape) \
                                            .to_event(self.bias.dim())

        self.weight_prior = weight_prior
        self.weight = PyroSample(prior=self.weight_prior)
        if bias:
            self.bias_prior = bias_prior
            self.bias = PyroSample(prior=self.bias_prior)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        weight = self.weight

        if weight.dim() == 2:
            return F.linear(input, weight, bias)
        else:
            if input.dim() == 2:
                input = input.unsqueeze(0).expand(weight.size(0), *input.size())

            return torch.baddbmm(bias.unsqueeze(1), input, weight.transpose(-1, -2))

    def to(self, *args, **kwargs):
        self.dist_to(self.weight_prior, *args, **kwargs)
        if self.bias is not None:
            self.dist_to(self.bias_prior, *args, **kwargs)

        return super().to(*args, *kwargs)

    def dist_to(self, dist, *args, **kwargs):
        for key, value in dist.__dict__.items():
            if torch.is_tensor(value):
                dist.__dict__[key] = value.to(*args, **kwargs)
            elif isinstance(value, distributions.Distribution):
                self.dist_to(value, *args, **kwargs)


class PyroSigmoid(nn.Sigmoid, PyroModule):
    pass


class PyroTanh(nn.Tanh, PyroModule):
    pass


class PyroReLU(nn.ReLU, PyroModule):
    pass


class BNNNotSampledError(Exception):
    pass


class PyroMCMCBNN(nn.Sequential, PyroModule):
    def __init__(self, *args, sigma=1.0, step_size=1e-6, num_steps=50):
        super().__init__(*args)

        self.sigma = sigma

        # self.hmc_kernel = HMC(self, step_size=step_size, num_steps=num_steps, jit_compile=True, ignore_jit_warnings=True)
        self.hmc_kernel = NUTS(self, step_size=step_size, full_mass=False, max_tree_depth=6, jit_compile=True, ignore_jit_warnings=True)
        self.mcmc = None

    def forward(self, X, y=None):
        mean = super().forward(X)

        if y is not None:
            with pyro.plate("data", device=X.device):
                obs = pyro.sample("obs", dist.Normal(mean, self.sigma), obs=y)

        return mean

    def sample(self, X, y, num_samples=200, reject=200):
        self.mcmc = MCMC(self.hmc_kernel, num_samples=num_samples, warmup_steps=reject)
        self.mcmc.run(X, y)

    def func_index(self, func, sample_indices, *args, **kwargs):
        if self.mcmc is None:
            raise BNNNotSampledError()

        sample_indices = self.clean_index(sample_indices, self.mcmc.num_samples).to(self._first_sample.device)
        samples = select_samples_by_idx(self.mcmc._samples, sample_indices)

        predictive = Predictive(func, posterior_samples=samples, return_sites=('_RETURN',), parallel=True)
        y = predictive(*args, **kwargs)['_RETURN']

        return y

    @property
    def _first_sample(self):
        return next(iter(self.mcmc._samples.values()))

    def clean_index(self, indices, length):
        def saturate(i):
            return min(max(i, -length), length - 1)

        if isinstance(indices, slice):
            start = saturate(indices.start)
            stop = None if indices.stop is None else saturate(indices.stop)

            indices = list(range(start, stop, indices.step))

        indices = torch.as_tensor(indices)
        if torch.any((indices < -length) | (indices >= length)):
            raise IndexError('sample index out of range')

        # Shift negative to positive
        return (indices + self.mcmc.num_samples) % self.mcmc.num_samples

    def predict_index(self, sample_indices, X):
        return self.func_index(self, sample_indices, X)

    def predict_dist(self, X, num_samples=None):
        if self.mcmc is None:
            raise BNNNotSampledError()

        predictive = Predictive(self, posterior_samples=self.mcmc.get_samples(num_samples), return_sites=('_RETURN',), parallel=True)
        y = predictive(X)['_RETURN']
        return y

    def predict_mean(self, X, num_samples=None):
        y = self.predict_dist(X, num_samples=num_samples)
        return y.mean(0)

    def to(self, *args, **kwargs):
        for module in self:
            module.to(*args, **kwargs)

        return super().to(*args, *kwargs)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict['samples'] = None if self.mcmc is None else self.mcmc._samples

        return state_dict

    def load_state_dict(self, state_dict: 'OrderedDict[str, Tensor]', strict: bool = True):
        samples = state_dict.pop('samples')
        if samples is None:
            logger.warning('No samples in loaded BNN')
        else:
            num_samples = next(iter(samples.values())).size(1)
            self.mcmc = MCMC(self.hmc_kernel, num_samples=num_samples, warmup_steps=0)
            self.mcmc._samples = samples

        super().load_state_dict(state_dict, strict)


def select_samples_by_idx(samples, sample_indices, group_by_chain=False):
    """
    Performs selection from given MCMC samples.

    :param dictionary samples: Samples object to sample from.
    :param IntTensor sample_indices: Indices of samples to return.
    :param bool group_by_chain: Whether to preserve the chain dimension. If True,
        all samples will have num_chains as the size of their leading dimension.
    :return: dictionary of samples keyed by site name.
    """
    if not samples:
        raise ValueError("No samples found from MCMC run.")
    if group_by_chain:
        batch_dim = 1
    else:
        samples = {k: v.reshape((-1,) + v.shape[2:]) for k, v in samples.items()}
        batch_dim = 0
    samples = {k: v.index_select(batch_dim, sample_indices) for k, v in samples.items()}
    return samples
