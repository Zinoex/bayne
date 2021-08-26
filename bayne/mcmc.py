import torch
from torch import nn

from bayne.container import ParameterQueue
from bayne.distributions import PriorWeightDistribution
from bayne.sampler import HamiltonianMonteCarlo
from bayne.util import ResetableModule


class MonteCarloBNN(nn.Module, ResetableModule):
    def __init__(self, network, sampler=HamiltonianMonteCarlo(step_size=1e-4, num_steps=50)):
        super().__init__()

        self.network = network
        self.num_states = 0
        self.sampler = sampler

        def _replace_parameter(m):
            for name, param in list(m.named_parameters(recurse=False)):
                del m._parameters[name]
                queue = ParameterQueue(param.data, maxlen=None)
                setattr(m, name, queue)
        self.apply(_replace_parameter)

    def sample(self, negative_log_prob, num_samples=200, reject=200, progress_bar=True):
        def _set_maxlen(m):
            if isinstance(m, ParameterQueue):
                m.maxlen = num_samples
        self.apply(_set_maxlen)

        self.sampler.sample(self, negative_log_prob, num_samples, reject, progress_bar)
        self.num_states = num_samples

    def forward(self, *args, state_idx=None, **kwargs):
        def _set_active_index(m):
            if isinstance(m, ParameterQueue):
                m.active_index = state_idx
        self.apply(_set_active_index)

        return self.network(*args, **kwargs)

    def predict_dist(self, *args, num_samples=None, dim=0, **kwargs):
        preds = [self(*args, **kwargs, state_idx=idx) for idx in range(self.num_states)]
        preds = torch.stack(preds, dim=dim)
        return preds

    def predict_mean(self, *args, num_samples=None, dim=0, **kwargs):
        preds = self.predict_dist(*args, dim=dim, **kwargs)
        return preds.mean(dim=dim)

    def log_prior(self, dist=PriorWeightDistribution()):
        return torch.stack([dist.log_prior(w).sum() for w in self.network.parameters()]).sum()

    def save(self):
        def _save(m):
            if isinstance(m, ParameterQueue):
                m.save()
        self.apply(_save)

    def zero_grad(self, set_to_none: bool = False) -> None:
        super().zero_grad(set_to_none)

        def _zero_grad_param_queue(m):
            if isinstance(m, ParameterQueue):
                for p in m.deque:
                    if p.grad is not None:
                        if set_to_none:
                            p.grad = None
                        else:
                            if p.grad.grad_fn is not None:
                                p.grad.detach_()
                            else:
                                p.grad.requires_grad_(False)
                            p.grad.zero_()
        self.apply(_zero_grad_param_queue)
