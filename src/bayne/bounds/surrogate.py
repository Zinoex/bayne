import torch
from torch import nn
import torch.nn.functional as F

from bayne.bounds.util import notnull


def surrogate_model(model):
    with notnull(getattr(model, '_pyro_context', None)):
        layers = []

        for module in model:
            with notnull(getattr(module, '_pyro_context', None)):
                if isinstance(module, nn.Linear):
                    layers.append(SurrogateLinear(module.weight, module.bias))
                else:
                    layers.append(module)

        return nn.Sequential(*layers)


class SurrogateLinear(nn.Module):
    def __init__(self, weight, bias):
        super().__init__()

        self.weight = weight
        self.bias = bias
        self.out_features = weight.size(-2)

    def forward(self, input):
        weight = self.weight
        bias = self.bias

        if weight.dim() == 2:
            return F.linear(input, weight, bias)
        else:
            if input.dim() == 2:
                input = input.unsqueeze(0).expand(weight.size(0), *input.size())

            return torch.baddbmm(bias.unsqueeze(1), input, weight.transpose(-1, -2))
