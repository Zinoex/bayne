import warnings

import torch
from torch import nn
from torch.nn.modules.dropout import _DropoutNd


class BaseMCDropout(nn.Module):
    def __init__(self):
        super().__init__()

        # Train = True ensures that dropout is enabled
        self.train()

    def assert_dropout(self):
        # Ensure that a dropout is available to perform the bayesian approximation
        def count_dropout(module):
            if isinstance(module, _DropoutNd):
                return 1
            else:
                counts = [count_dropout(submodule) for submodule in module.children()]
                return sum(counts)

        num_dropout = count_dropout(self)
        assert num_dropout > 0

    def predict_dist(self, *args, num_samples=1, dim=0, **kwargs):
        # Train = True ensures that dropout is enabled
        self.train()
        self.assert_dropout()

        preds = torch.stack([self(*args, **kwargs) for _ in range(num_samples)], dim=dim)
        return preds

    def predict_mean(self, *args, num_samples=1, dim=0, **kwargs):
        preds = self.predict_dist(*args, num_samples=num_samples, dim=dim, **kwargs)
        return preds.mean(dim=dim)

    def train(self, mode=True):
        # Override train to ensure dropout is always enabled
        if not mode:
            warnings.warn('Dropout must be enabled to provide uncertainty estimates; even during testing.')

        super(BaseMCDropout, self).train(True)
