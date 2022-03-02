from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from bayne.bounds import ibp, crown_ibp, crown
from bayne.mcmc import PyroMCMCBNN, PyroBatchLinear, PyroTanh
from examples.noisy_sine import NoisySineDataset
from examples.test import test


################################################################
# An example of Markov chain Monte Carlo BNN for regression on
# a noisy sine dataset
################################################################


def train(model, args):
    dataset = NoisySineDataset(dim=args.dim, train_size=2**12)

    X, y = dataset[:]
    X, y = X.to(args.device), y.to(args.device)

    model.sample(X, y, num_samples=50, reject=200)


def main(args):
    net = crown(crown_ibp(ibp(PyroMCMCBNN(
            PyroBatchLinear(args.dim, 64),
            PyroTanh(),
            PyroBatchLinear(64, 64),
            PyroTanh(),
            PyroBatchLinear(64, 1),
            sigma=0.2
    )))).to(args.device)

    train(net, args)
    net.state_dict()
    test(net, args)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--device', choices=list(map(torch.device, ['cuda', 'cpu'])), type=torch.device, default='cuda',
                        help='Select device for tensor operations')
    parser.add_argument('--dim', choices=[1, 2], type=int, default=1, help='Dimensionality of the noisy sine')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
