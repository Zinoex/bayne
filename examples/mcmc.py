import torch
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from bayne.bounds.ibp import SampleIntervalBoundPropagation
from bayne.mcmc import MonteCarloBNN
from bayne.nll import GaussianNegativeLogProb, MinibatchGaussianNegativeLogProb
from bayne.sampler import HamiltonianMonteCarlo, StochasticGradientHMC
from bayne.util import timer
from examples.noisy_sine import NoisySineDataset
from examples.test import test


################################################################
# An example of Markov chain Monte Carlo BNN for regression on
# a noisy sine dataset
################################################################


class ExampleMonteCarloBNN(nn.Sequential):
    def __init__(self, in_features, out_features):
        super().__init__(
            Linear(in_features, 128),
            nn.Tanh(),
            Linear(128, 64),
            nn.Tanh(),
            Linear(64, out_features)
        )


def train(model):
    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    negative_log_prob = MinibatchGaussianNegativeLogProb(model, dataloader)

    model.sample(negative_log_prob, num_samples=1000, reject=20)


def main():
    subnetwork = ExampleMonteCarloBNN(1, 1)
    model = MonteCarloBNN(subnetwork, sampler=StochasticGradientHMC(step_size=1e-6))
    train(model)
    test(model, 'HMC')


if __name__ == '__main__':
    main()
