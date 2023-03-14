from argparse import ArgumentParser

import torch
from torch.nn import MSELoss, BCEWithLogitsLoss

from bayne.mcmc import NormalLikelihoodMCMCBNN, PyroSequential, PyroBatchLinear, PyroTanh,\
    BernoulliLogitsLikelihoodMCMCBNN
from examples.noisy_sine import NoisySineDataset
from examples.test import test
from examples.weather import WeatherHistoryDataset


################################################################
# An example of Markov chain Monte Carlo BNN for regression on
# a noisy sine dataset
################################################################


def train(model, dataset, args):
    X, y = dataset[:]
    X, y = X.to(args.device), y.to(args.device)

    model.sample(X, y, num_samples=100, reject=50)


def main(args):
    if args.dataset == 'noisy-sine':
        dataset = NoisySineDataset(dim=args.dim, train_size=2**10)

        model = PyroSequential(
                PyroBatchLinear(args.dim, 64),
                PyroTanh(),
                PyroBatchLinear(64, 64),
                PyroTanh(),
                PyroBatchLinear(64, 1))

        net = NormalLikelihoodMCMCBNN(model, sigma=0.4).to(args.device)
        criterion = MSELoss()
    elif args.dataset == 'weather-forecast':
        dataset = WeatherHistoryDataset()

        model = PyroSequential(
                PyroBatchLinear(2, 64),
                PyroTanh(),
                PyroBatchLinear(64, 64),
                PyroTanh(),
                PyroBatchLinear(64, 1))

        net = BernoulliLogitsLikelihoodMCMCBNN(model).to(args.device)
        criterion = BCEWithLogitsLoss()
    else:
        raise ValueError('Dataset not found')

    train(net, dataset, args)
    net.state_dict()
    test(net, dataset, criterion, args)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cuda', help='Select device for tensor operations')
    parser.add_argument('--dataset', choices=['noisy-sine', 'weather-forecast'], type=str, default='noisy-sine', help='Dataset')
    parser.add_argument('--dim', type=int, default=1, help='Dimensionality of the input')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
