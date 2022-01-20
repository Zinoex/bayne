from typing import Callable

import torch
from torch import nn

from bayne.bounds.crown import alpha_beta_sequential, linear_bounds, surrogate_model
from bayne.bounds.util import notnull, add_method


def crown_ibp(model):
    @torch.no_grad()
    def crown_ibp(self: nn.Sequential, lower, upper):
        with notnull(getattr(self, '_pyro_context', None)):
            batch_size = lower.size(0)

            LBs, UBs = self.ibp(lower, upper, pre=True)

            alpha, beta = alpha_beta_sequential(self, LBs, UBs)
            bounds = linear_bounds(self, alpha, beta, batch_size, lower.device)

            return bounds

    add_method(model, 'crown_ibp', crown_ibp)

    return model
