import torch
from torch import nn


class MonteCarloBNN(nn.Module):
    def __init__(self, network):
        super().__init__()

        self.network = network
        self.states = []

    def forward(self, *args, state_idx=0, **kwargs):
        self.network.load_state_dict(self.states[state_idx])

        return self.network(*args, **kwargs)

    def predict_dist(self, *args, dim=0, **kwargs):
        preds = [self(*args, **kwargs, state_idx=idx) for idx in range(len(self.states))]
        preds = torch.stack(preds, dim=dim)
        return preds

    def predict_mean(self, *args, num_samples=1, dim=0, **kwargs):
        preds = self.predict_dist(*args, num_samples=num_samples, dim=dim, **kwargs)
        return preds.mean(dim=dim)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict['states'] = self.states

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        states = state_dict.pop('states')

        super(MonteCarloBNN, self).load_state_dict(state_dict, strict)
