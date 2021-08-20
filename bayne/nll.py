import abc
import time

import torch
from torch import distributions

from bayne.util import TensorList


class NegativeLogProb(abc.ABC):
    @abc.abstractmethod
    def dVdq(self):
        raise NotImplementedError()


class GaussianNegativeLogProb(NegativeLogProb):
    def __init__(self, network, X, y, noise=0.1, log_nll=True):
        self.network = network
        self.X, self.y = X, y
        self.dist = distributions.Normal(self.y, noise)

        self.log_nll = log_nll
        self.last_log = time.time()

    def __call__(self, *args, **kwargs):
        y_pred = self.network(self.X)

        neg_log_prob = -self.dist.log_prob(y_pred).sum()
        neg_log_prior = -self.network.log_prior()

        self.log(neg_log_prob, neg_log_prior)

        return neg_log_prob + neg_log_prior

    def log(self, nll, nlp):
        now = time.time()
        if self.log_nll and now - self.last_log > 1.0:  # 1s
            print(f'NLL: {nll.item()}, NLP: {nlp.item()}')
            self.last_log = now

    @torch.enable_grad()
    def dVdq(self):
        output = self()
        q0 = self.network.parameters()

        return torch.autograd.grad(output, q0)


class MinibatchNegativeLogProb(NegativeLogProb, abc.ABC):
    pass


class MinibatchGaussianNegativeLogProb(MinibatchNegativeLogProb):
    def __init__(self, network, dataloader, noise=0.1, log_nll=True):
        self.network = network
        self.dataloader = dataloader
        self.num_batches = len(self.dataloader)
        self.iter = iter(self.dataloader)
        self.noise = noise

        self.log_nll = log_nll
        self.last_log = time.time()

    def __call__(self, *args, **kwargs):
        try:
            X, y = next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            X, y = next(self.iter)

        y_pred = self.network(X)
        dist = distributions.Normal(y, self.noise)

        neg_log_prob = -self.num_batches * dist.log_prob(y_pred).sum()
        neg_log_prior = -self.network.log_prior()

        self.log(neg_log_prob, neg_log_prior)

        return neg_log_prob + neg_log_prior

    def log(self, nll, nlp):
        now = time.time()
        if self.log_nll and now - self.last_log > 1.0:  # 1s
            print(f'NLL: {nll.item()}, NLP: {nlp.item()}')
            self.last_log = now

    @torch.enable_grad()
    def dVdq(self):
        output = self()
        q0 = self.network.parameters()

        return torch.autograd.grad(output, q0)
