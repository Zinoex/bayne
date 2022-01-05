import torch
import torch.nn.functional as F

from torch import nn

from bayne.bounds.util import notnull, add_method


def interval_bound_propagation(class_or_obj):
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
        return interval_bound_propagation_sequential(class_or_obj)
    elif isinstance(class_or_obj, nn.Linear) or (isinstance(class_or_obj, type) and issubclass(class_or_obj, nn.Linear)):
        return interval_bound_propagation_linear(class_or_obj)
    else:
        return interval_bound_propagation_activation(class_or_obj)


def interval_bound_propagation_sequential(class_or_obj):
    @torch.no_grad()
    def ibp(self: nn.Sequential, lower: torch.Tensor, upper: torch.Tensor, pre: bool = False):
        with notnull(getattr(self, '_pyro_context', None)):
            LBs, UBs = [], []

            for i, module in enumerate(self):
                if not hasattr(module, 'ibp'):
                    # Decorator also adds the method inplace.
                    interval_bound_propagation(module)

                if pre:
                    LBs.append(lower)
                    UBs.append(upper)

                lower, upper = module.ibp(lower, upper)

            if pre:
                return LBs, UBs
            else:
                return lower, upper

    add_method(class_or_obj, 'ibp', ibp)
    return class_or_obj


def interval_bound_propagation_linear(class_or_obj):
    @torch.no_grad()
    def ibp(self: nn.Linear, lower: torch.Tensor, upper: torch.Tensor):
        with notnull(getattr(self, '_pyro_context', None)):
            mid = (lower + upper) / 2
            diff = (upper - lower) / 2

            weight = self.weight
            abs_weight = torch.abs(weight)
            bias = self.bias

            if weight.dim() == 2:
                w_mid = F.linear(mid, weight)
                w_diff = F.linear(diff, abs_weight)
            else:
                if mid.dim() == 2:
                    mid = mid.unsqueeze(0).expand(weight.size(0), *mid.size())
                    diff = diff.unsqueeze(0).expand(weight.size(0), *diff.size())

                w_mid = torch.bmm(mid, weight.transpose(-1, -2))
                w_diff = torch.bmm(diff, abs_weight.transpose(-1, -2))

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


def interval_bound_propagation_activation(class_or_obj):
    @torch.no_grad()
    def ibp(self: nn.Module, lower: torch.Tensor, upper: torch.Tensor):
        with notnull(getattr(self, '_pyro_context', None)):
            return self(lower), self(upper)

    add_method(class_or_obj, 'ibp', ibp)
    return class_or_obj
