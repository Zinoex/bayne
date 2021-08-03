import torch
from torch import nn
from torch.nn.modules.dropout import _DropoutNd


class BaseMCDropout(nn.Module):
    def __init__(self):
        super().__init__()

        # Ensure that a dropout is available to perform the bayesian approximation
        def count_dropout(module):
            if isinstance(module, _DropoutNd):
                return 1
            else:
                counts = [count_dropout(submodule) for submodule in module.modules()]
                return sum(counts)

        num_dropout = count_dropout(self)
        assert num_dropout > 0

    def predict_dist(self, *args, num_samples=1, dim=None, **kwargs):
        preds = torch.stack([self(*args, **kwargs) for _ in range(num_samples)], dim=dim)
        return preds

    def predict_mean(self, *args, num_samples=1, dim=None, **kwargs):
        preds = self.predict_dist(*args, num_samples=num_samples, dim=dim, **kwargs)
        return preds.mean(0)


