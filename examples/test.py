import torch
from matplotlib import pyplot as plt
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from examples.weather import WeatherHistoryDataset, y_scaler


@torch.no_grad()
def test(model):
    num_samples = 1000

    criterion = MSELoss()

    dataset = WeatherHistoryDataset(train=False)

    # Average MSE test
    dataloader = DataLoader(dataset, batch_size=len(dataset), num_workers=0)
    X, y = next(iter(dataloader))

    y_pred = model.predict_mean(X, num_samples=num_samples)
    avg_loss = criterion(y_pred, y)

    print(f'Average loss: {avg_loss.item()}')

    # Distribution test
    dataloader = DataLoader(dataset, num_workers=0)
    X, y = next(iter(dataloader))

    y_dist = model.predict_dist(X, num_samples=num_samples)
    y_dist = y_scaler.inverse_transform(y_dist.numpy())[:, 0]

    plt.hist(y_dist)
    plt.axvline(y_scaler.inverse_transform(y), color='r', linestyle='dashed', linewidth=1)
    plt.show()
