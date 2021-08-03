import torch
from torch import nn, optim, distributions
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from bnn.variational_inference import BaseVariationalBNN, VariationalLinear
from examples.noisy_sine import NoisySineDataset
from examples.test import test


################################################################
# An example of variational inference BNN for regression on
# a noisy sine dataset
################################################################


class ExampleVariationalBNN(BaseVariationalBNN):
    def __init__(self, in_features):
        super().__init__()

        self.model = nn.Sequential(
            VariationalLinear(in_features, 20),
            nn.ReLU(),
            VariationalLinear(20, 20),
            nn.ReLU(),
            VariationalLinear(20, 1)
        )

    def forward(self, x):
        return self.model(x)


def train(model):
    num_epochs = 100
    optimizer = optim.Adam(model.parameters(), lr=0.08)

    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=0)

    num_batches = len(dataloader)

    for epoch in trange(num_epochs, desc='Epoch'):
        for X, y in tqdm(dataloader, desc='Iteration'):
            optimizer.zero_grad(set_to_none=True)
            model.zero_kl()

            y_pred = model(X)
            sigma = torch.full_like(y_pred, 1.0)
            dist = distributions.Normal(y_pred, sigma)
            loss = -dist.log_prob(y).sum() + model.kl_loss() / num_batches
            loss.backward()

            optimizer.step()

        print(f'Loss: {loss.item()}')


def main():
    model = ExampleVariationalBNN(1)
    train(model)
    test(model)


if __name__ == '__main__':
    main()
