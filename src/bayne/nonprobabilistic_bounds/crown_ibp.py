from torch import nn

from bayne.bounds.ibp import ibp
from bayne.bounds.crown import alpha_beta_sequential, linear_bounds
from bayne.bounds.util import add_method


def crown_ibp(model):
    def crown_ibp(self: nn.Sequential, lower, upper):
        ibp(self)

        batch_size = lower.size(0)

        LBs, UBs = self.ibp(lower, upper, pre=True)

        alpha, beta = alpha_beta_sequential(self, LBs, UBs)
        bounds = linear_bounds(self, alpha, beta, batch_size, lower.device)

        return bounds

    add_method(model, 'crown_ibp', crown_ibp)

    return model
