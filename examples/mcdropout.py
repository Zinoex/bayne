from torch import nn, optim
from torch.nn import MSELoss
from torch.utils.data import DataLoader
from tqdm import trange, tqdm

from bnn.mcdropout import BaseMCDropout
from examples.noisy_sine import NoisySineDataset
from examples.test import test


################################################################
# An example of Monte Carlo Dropout for regression on
# a noisy sine dataset
################################################################


class ExampleMCDropout(BaseMCDropout):
    def __init__(self, in_features, alpha=0.2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Dropout(alpha),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(alpha),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(alpha),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        self.assert_dropout()
        return self.model(x)


def train(model):
    num_epochs = 100
    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters())

    dataset = NoisySineDataset()
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)

    for epoch in trange(num_epochs, desc='Epoch'):
        for X, y in tqdm(dataloader, desc='Iteration'):
            optimizer.zero_grad(set_to_none=True)

            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

        print(f'Loss: {loss.item()}')


def main():
    model = ExampleMCDropout(1)
    train(model)
    test(model)


if __name__ == '__main__':
    main()
