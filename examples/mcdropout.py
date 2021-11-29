from argparse import ArgumentParser

import torch
from torch import nn, optim, distributions
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from bayne.mcdropout import BaseMCDropout
from examples.noisy_sine import NoisySineDataset
from examples.test import test


################################################################
# An example of Monte Carlo Dropout for regression on
# a noisy sine dataset
################################################################


class ExampleMCDropout(BaseMCDropout):
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__(
            nn.Dropout(alpha),
            nn.Linear(in_features, 128),
            nn.Tanh(),
            nn.Dropout(alpha),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(alpha),
            nn.Linear(64, out_features)
        )


def train(model, device):
    num_epochs = 1000
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters())

    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    for epoch in trange(num_epochs, desc='Epoch'):
        for X, y in tqdm(dataloader, desc='Iteration'):
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)

            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

        print(f'Loss: {loss.item()}')


def main(args):
    device = torch.device(args.device)

    model = ExampleMCDropout(1, 1).to(device)
    train(model, device)
    test(model, device, 'MCD')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda', help='Select device for tensor operations')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
