from torch import nn, optim, distributions
from torch.nn import Linear
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from bnn.mcmc import MonteCarloBNN
from bnn.util import set_random_seed
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

    def nll():
        y_pred = model(X)

        dist = distributions.Normal(y, 0.025)
        neg_log_prob = -dist.log_prob(y_pred).mean()

        print(f'NLL: {neg_log_prob.item()}')
        return neg_log_prob

    model.sample(nll, reject=10)


def main():
    subnetwork = ExampleMonteCarloBNN(1, 1)
    model = MonteCarloBNN(subnetwork)
    train(model)
    test(model)


if __name__ == '__main__':
    main()
