import functools
import math

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

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
        def _set_maxlen_and_active_index(m):
            if isinstance(m, ParameterQueue):
                m.maxlen = num_samples
                m.active_index = None
        self.apply(_set_maxlen_and_active_index)

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


class BatchModule(nn.Module):
    def __init__(self):
        super(BatchModule, self).__init__()

        self.full = False

    @staticmethod
    def set_full(module):
        module.full = True

    @staticmethod
    def reset_full(module):
        module.full = False


class BatchModeFull:
    def __init__(self, network):
        self.network = network

    def __enter__(self):
        self.network.apply(BatchModule.set_full)

    def __exit__(self, type, value, traceback):
        self.network.apply(BatchModule.reset_full)


class BatchLinear(BatchModule):
        __constants__ = ['in_features', 'out_features']
        in_features: int
        out_features: int

        def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
            factory_kwargs = {'device': device, 'dtype': dtype}
            super(BatchLinear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = ParameterQueue(torch.empty((out_features, in_features), **factory_kwargs))
            if bias:
                self.bias = ParameterQueue(torch.empty(out_features, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)

        def forward(self, input: Tensor) -> Tensor:
            if self.weight.active_index is None or isinstance(self.weight.active_index, int):
                return F.linear(input, self.weight, self.bias)

            weight = self.weight.active_weights()
            if self.full:
                if input.dim() != 3:
                    input = input.unsqueeze(0).expand(weight.size(0), -1, -1)
                    input = input.transpose(-1, -2)
            else:
                input = input.unsqueeze(-1)

            assert input.size(0) == weight.size(0), 'Incompatible size for batch linear layer'

            if self.bias is None:
                res = torch.bmm(input, weight)
            else:
                bias = self.bias.active_weights()
                res = torch.baddbmm(bias.unsqueeze(-1), weight, input)

            if self.full:
                res = res.transpose(-1, -2)
            else:
                res = res.squeeze(-1)

            return res

        def extra_repr(self) -> str:
            return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None
            )


class BatchMonteCarloBNN(nn.Module, ResetableModule):
    def __init__(self, network, sampler=HamiltonianMonteCarlo(step_size=1e-4, num_steps=50)):
        super().__init__()

        self.network = network
        self.num_states = 0
        self.sampler = sampler

    def sample(self, negative_log_prob, num_samples=200, reject=200, progress_bar=True):
        def _set_maxlen_and_active_index(m):
            if isinstance(m, ParameterQueue):
                m.maxlen = num_samples
                m.active_index = None
        self.apply(_set_maxlen_and_active_index)

        self.sampler.sample(self, negative_log_prob, num_samples, reject, progress_bar)
        self.num_states = num_samples

    def forward(self, *args, state_indices=None, **kwargs):
        def _set_active_index(m):
            if isinstance(m, ParameterQueue):
                m.active_index = state_indices
        self.apply(_set_active_index)

        return self.network(*args, **kwargs)

    def predict_dist(self, *args, num_samples=None, dim=0, **kwargs):
        with BatchModeFull(self):
            return self(*args, **kwargs, state_indices=torch.arange(self.num_states))

    def predict_mean(self, *args, num_samples=None, dim=0, **kwargs):
        preds = self.predict_dist(*args, dim=dim, **kwargs)
        return preds.mean(dim=0)

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
