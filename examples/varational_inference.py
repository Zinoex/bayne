from torch import nn, optim, distributions
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from bayne.util import set_random_seed
from bayne.variational_inference import BaseVariationalBNN, VariationalLinear
from examples.noisy_sine import NoisySineDataset
from examples.test import test


################################################################
# An example of variational inference BNN for regression on
# a noisy sine dataset
################################################################


class ExampleVariationalBNN(BaseVariationalBNN):
    def __init__(self, in_features, out_features):
        super().__init__(
            VariationalLinear(in_features, 128),
            nn.Tanh(),
            VariationalLinear(128, 64),
            nn.Tanh(),
            VariationalLinear(64, out_features)
        )


def train(model):
    num_epochs = 1000
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    num_batches = len(dataloader)

    def criterion(output, target):
        dist = distributions.Normal(target, 0.1)
        return -dist.log_prob(output).sum()

    for epoch in trange(num_epochs, desc='Epoch'):
        for idx, (X, y) in enumerate(tqdm(dataloader, desc='Iteration')):
            optimizer.zero_grad(set_to_none=True)

            effective_idx = epoch * num_batches + idx
            loss = model.sample_elbo(effective_idx, X, y, criterion, num_samples=3, num_batches=num_batches)
            loss.backward()

            optimizer.step()

        print(f'Loss: {loss.item()}')


def main():
    model = ExampleVariationalBNN(1, 1)
    train(model)
    test(model, 'VI')


if __name__ == '__main__':
    main()
