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

    def interval_bounds(self, model: nn.Sequential, input_bounds: Tuple[torch.Tensor, torch.Tensor]):
        lower, upper = input_bounds

        for module in model:
            if isinstance(module, (nn.Linear, VariationalLinear)):
                first = module(lower)
                second = module(upper)

                lower = torch.min(first, second)
                upper = torch.max(first, second)
            else:
                lower = module(lower)
                upper = module(upper)

        return lower, upper

    def linear_bounds(self, model: nn.Sequential, input_bounds: Tuple[torch.Tensor, torch.Tensor]):
        raise NotImplementedError()
