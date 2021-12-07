import math
from argparse import ArgumentParser

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pyro.nn import PyroModule, PyroSample
from pyro.infer.mcmc import HMC, MCMC
from pyro.infer import Predictive
import pyro.distributions as dist
import pyro

from examples.noisy_sine import NoisySineDataset
from examples.test import test


class PyroBatchLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        weight = self.weight

        if bias.dim() == 1:
            return F.linear(input, weight, bias)
        else:
            if input.dim() == 2:
                input = input.unsqueeze(0).expand(weight.size(0), *input.size())

            return torch.baddbmm(bias.unsqueeze(1), input, weight.transpose(-1, -2))


class BNN(nn.Sequential, PyroModule):
    def __init__(self, in_features, out_features, sigma, device=None):
        super().__init__(
            PyroModule[PyroBatchLinear](in_features, 16),
            PyroModule[nn.Tanh](),
            PyroModule[PyroBatchLinear](16, 16),
            PyroModule[nn.Tanh](),
            PyroModule[PyroBatchLinear](16, 16),
            PyroModule[nn.Tanh](),
            PyroModule[PyroBatchLinear](16, out_features),
        )

        for m in self.modules():
            for name, value in list(m.named_parameters(recurse=False)):
                setattr(m, name, PyroSample(prior=dist.Normal(torch.as_tensor(0.0, device=device), torch.as_tensor(1.0, device=device))
                                            .expand(value.shape)
                                            .to_event(value.dim())))

        self.sigma = sigma

        self.hmc_kernel = HMC(self, step_size=1e-6, num_steps=10, jit_compile=True, ignore_jit_warnings=True)
        self.mcmc = MCMC(self.hmc_kernel, num_samples=3000, warmup_steps=1000)

    def forward(self, X, y=None):
        mean = super().forward(X)

        if y is not None:
            with pyro.plate("data"):
                obs = pyro.sample("obs", dist.Normal(mean, self.sigma), obs=y)

        return mean

    def sample(self, X, y):
        self.mcmc.run(X, y)

    def predict_dist(self, X, num_samples=None):
        predictive = Predictive(self, posterior_samples=self.mcmc.get_samples(num_samples), return_sites=('_RETURN',), parallel=True)
        y = predictive(X)['_RETURN']
        return y

    def predict_mean(self, X, num_samples=None):
        y = self.predict_dist(X, num_samples=num_samples)
        return y.mean(0)

################################################################
# An example of Markov chain Monte Carlo BNN for regression on
# a noisy sine dataset
################################################################


def train(model, device):
    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

    X, y = next(iter(dataloader))
    X, y = X.to(device), y.to(device)
    model.sample(X, y)


def main(args):
    device = torch.device(args.device)

    net = BNN(1, 1, sigma=0.2, device=device)

    train(net, device)
    test(net, device, 'HMC')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Select device for tensor operations')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
