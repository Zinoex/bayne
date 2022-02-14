from typing import Union, Optional

import torch
import torch.nn.functional as F

from torch import nn

from .surrogate import SurrogateLinear
from .util import notnull, add_method, IntervalBounds, LayerBounds


def ibp(class_or_obj):
    """
    Propagate a sample interval through a deterministic fully-connected feedforward network.
    This can correspond to a weight sample using MCMC.

    Assumptions:
    - The network is nn.Sequential
    - All layers of the network is nn.Linear or a monotone activation function

    This method does _not_ support weight bounds, only input bounds.

    @misc{wicker2020probabilistic,
      title={Probabilistic Safety for Bayesian Neural Networks},
      author={Matthew Wicker and Luca Laurenti and Andrea Patane and Marta Kwiatkowska},
      year={2020},
      eprint={2004.10281},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
    """

    if isinstance(class_or_obj, nn.Sequential) or (isinstance(class_or_obj, type) and issubclass(class_or_obj, nn.Sequential)):
        return ibp_sequential(class_or_obj)
    elif isinstance(class_or_obj, (nn.Linear, SurrogateLinear)) or \
            (isinstance(class_or_obj, type) and issubclass(class_or_obj, (nn.Linear, SurrogateLinear))):
        return ibp_linear(class_or_obj)
    else:
        return ibp_activation(class_or_obj)


def ibp_sequential(class_or_obj):
    def _ibp(self: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor, pre: bool = False) -> Union[IntervalBounds, LayerBounds]:
        with notnull(getattr(self, '_pyro_context', None)):
            layer_bounds = []

            for i, module in enumerate(self):
                if not hasattr(module, 'ibp'):
                    # Decorator also adds the method inplace.
                    ibp(module)

                if pre:
                    layer_bounds.append((lower, upper))

                lower, upper = module.ibp(lower, upper)

            if pre:
                return layer_bounds
            else:
                return lower, upper

    add_method(class_or_obj, 'ibp', _ibp)
    return class_or_obj


def ibp_linear(class_or_obj):
    @torch.no_grad()
    def ibp(self: nn.Linear, lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
        with notnull(getattr(self, '_pyro_context', None)):
            mid = (lower + upper) / 2
            diff = (upper - lower) / 2

            weight = self.weight
            abs_weight = weight.abs()
            bias = self.bias

            if weight.dim() == 2:
                w_mid = F.linear(mid, weight)
                w_diff = F.linear(diff, abs_weight)
            else:
                w_mid = torch.matmul(weight.unsqueeze(1), mid.unsqueeze(-1))[..., 0]
                w_diff = torch.matmul(abs_weight.unsqueeze(1), diff.unsqueeze(-1))[..., 0]

            lower = w_mid - w_diff
            upper = w_mid + w_diff

            if bias is not None:
                if bias.dim() == 2:
                    bias = bias.unsqueeze(-2)

                lower += bias
                upper += bias

            return lower, upper

    add_method(class_or_obj, 'ibp', ibp)
    return class_or_obj


def ibp_activation(class_or_obj):
    @torch.no_grad()
    def ibp(self: nn.Module, lower: torch.Tensor, upper: torch.Tensor) -> IntervalBounds:
        with notnull(getattr(self, '_pyro_context', None)):
            return self(lower), self(upper)

    add_method(class_or_obj, 'ibp', ibp)
    return class_or_obj
