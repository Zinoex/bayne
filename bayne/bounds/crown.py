import torch

from bayne.bounds.core import Bounds


class CROWN(Bounds):
    @torch.no_grad()
    def interval_bounds(self, model, input_bounds):
        pass

    @torch.no_grad()
    def linear_bounds(self, model, input_bounds):
        pass
