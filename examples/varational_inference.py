from argparse import ArgumentParser

import torch
from torch import nn, optim, distributions
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

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


def train(model, device):
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
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            effective_idx = epoch * num_batches + idx
            loss = model.sample_elbo(effective_idx, X, y, criterion, num_samples=3, num_batches=num_batches)
            loss.backward()

            optimizer.step()

        print(f'Loss: {loss.item()}')


def main(args):
    device = torch.device(args.device)

    model = ExampleVariationalBNN(1, 1).to(device)
    train(model, device)
    test(model, device, 'VI')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Select device for tensor operations')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
