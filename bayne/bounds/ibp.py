from typing import Tuple

import torch

from bayne.bounds.core import Bounds
from torch import nn

from bayne.variational_inference import VariationalLinear


class SampleIntervalBoundPropagation(Bounds):
    """
    Propagate a sample interval through a deterministic fully-connected feedforward network.
    This can correspond to a weight sample using MCMC.

    Assumptions:
    - The network is nn.Sequential
    - All layers of the network is nn.Linear or a monotone activation function

    This method does _not_ support weight bounds.

    @misc{wicker2020probabilistic,
      title={Probabilistic Safety for Bayesian Neural Networks},
      author={Matthew Wicker and Luca Laurenti and Andrea Patane and Marta Kwiatkowska},
      year={2020},
      eprint={2004.10281},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
    """

    @torch.no_grad()
    def interval_bounds(self, model: nn.Sequential, input_bounds: Tuple[torch.Tensor, torch.Tensor], include_intermediate=False):
        lower, upper = input_bounds

        LBs, UBs = [], []

        for module in model:
            if isinstance(module, (nn.Linear, VariationalLinear)):
                mid = (lower + upper) / 2
                diff = (upper - lower) / 2

                w_mid = torch.matmul(module.weight, mid.unsqueeze(-1))[..., 0]
                w_diff = torch.matmul(torch.abs(module.weight), diff.unsqueeze(-1))[..., 0]

                lower = w_mid - w_diff
                upper = w_mid + w_diff

                if module.bias:
                    lower += module.bias
                    upper += module.bias
            else:
                lower = module(lower)
                upper = module(upper)

            LBs.append(lower)
            UBs.append(upper)

        if include_intermediate:
            return LBs, UBs
        else:
            return lower, upper

    @torch.no_grad()
    def linear_bounds(self, model: nn.Sequential, input_bounds: Tuple[torch.Tensor, torch.Tensor]):
        raise NotImplementedError()
