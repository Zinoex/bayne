from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

from bayne.mcmc import MonteCarloBNN
from bayne.nll import GaussianNegativeLogProb
from bayne.sampler import HamiltonianMonteCarlo
from examples.noisy_sine import NoisySineDataset
from examples.test import test


################################################################
# An example of Markov chain Monte Carlo BNN for regression on
# a noisy sine dataset
################################################################


class ExampleMonteCarloBNN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.model = nn.Sequential(
            Linear(in_features, 128),
            nn.SiLU(),
            Linear(128, 64),
            nn.SiLU(),
            Linear(64, out_features)
        )

    def forward(self, x):
        return self.model(x)


def train(model):
    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, num_workers=0)
    X, y = next(iter(dataloader))
    negative_log_prob = GaussianNegativeLogProb(model, X, y)

    model.sample(negative_log_prob, num_samples=1000, reject=20)


def main():
    subnetwork = ExampleMonteCarloBNN(1, 1)
    model = MonteCarloBNN(subnetwork, sampler=HamiltonianMonteCarlo(step_size=5e-4))
    train(model)
    test(model, 'HMC')


if __name__ == '__main__':
    main()
