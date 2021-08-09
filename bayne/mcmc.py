import copy

import torch
from torch import nn, distributions
from tqdm import trange

from bayne.distributions import PriorWeightDistribution
from bayne.sampler import HamiltonianMonteCarlo


class MonteCarloBNN(nn.Module):
    def __init__(self, network, step_size=0.0005, num_steps=100):
        super().__init__()

        self.network = network
        self.sampler = HamiltonianMonteCarlo(network.parameters(), step_size=step_size, num_steps=num_steps)
        self.states = []

    def sample(self, negative_log_prob, num_samples=1000, reject=0, progress_bar=True):
        self.states = []
        num_accept = 0

        r = trange if progress_bar else range

        for idx in r(num_samples + reject):
            if idx >= 0:
                self.states.append(copy.deepcopy(self.network.state_dict()))

            accept = self.sampler.step(negative_log_prob)
            if accept:
                num_accept += 1

        print(f'Acceptance ratio: {num_accept / (num_samples + reject)}')

    def forward(self, *args, state_idx=None, **kwargs):
        if state_idx is not None:
            self.network.load_state_dict(self.states[state_idx])

        return self.network(*args, **kwargs)

    def predict_dist(self, *args, num_samples=None, dim=0, **kwargs):
        preds = [self(*args, **kwargs, state_idx=idx) for idx in range(len(self.states))]
        preds = torch.stack(preds, dim=dim)
        return preds

    def predict_mean(self, *args, num_samples=None, dim=0, **kwargs):
        preds = self.predict_dist(*args, dim=dim, **kwargs)
        return preds.mean(dim=dim)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict['states'] = self.states

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.states = state_dict.pop('states')
        self.network.load_state_dict(self.states[0])

        super().load_state_dict(state_dict, strict)

    def log_prior(self, dist=PriorWeightDistribution()):
        return torch.stack([dist.log_prior(w).sum() for w in self.network.parameters()]).sum()
