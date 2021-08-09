import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions


class PosteriorWeightDistribution(nn.Module):
    def __init__(self, mu, rho):
        super().__init__()

        self.mu = mu
        self.rho = rho
        self.register_buffer('eps_w', torch.Tensor(self.mu.shape))
        self.sigma = None

    def sample(self):
        self.eps_w.data.normal_()
        # Apply sigma = log(1 + exp(rho)) to allow free parameterization to positive variance, R -> R+
        self.sigma = F.softplus(self.rho, beta=1, threshold=20)
        w = self.mu + self.sigma * self.eps_w
        return w

    def log_posterior(self, w):
        dist = distributions.Normal(self.mu, self.sigma)

        return dist.log_prob(w).sum()


class PriorWeightDistribution(nn.Module):
    def __init__(self, pi=1, sigma1=1, sigma2=0.1):
        super().__init__()

        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dist1 = torch.distributions.Normal(0, sigma1)
        self.dist2 = torch.distributions.Normal(0, sigma2)

    def log_prior(self, w):
        prob_n1 = torch.exp(self.dist1.log_prob(w))
        prob_n2 = torch.exp(self.dist2.log_prob(w))

        prior_pdf = (self.pi * prob_n1 + (1 - self.pi) * prob_n2)

        return torch.log(prior_pdf).sum()
