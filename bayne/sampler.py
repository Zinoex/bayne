import copy

import numpy as np
import torch
from torch import optim, distributions
from tqdm import trange

from bayne.nll import NegativeLogProb
from bayne.util import TensorList


class HamiltonianMonteCarlo:
    def __init__(self, step_size=0.0005, num_steps=100):
        self.step_size = step_size
        self.num_steps = num_steps
        self.kinetic_energy_type = 'log_prob_sum'  # Choices: log_prob_sum, dot_product

    def sample(self, network, negative_log_prob, num_samples=1000, reject=0, progress_bar=True):
        states = []
        num_accept = 0

        r = trange if progress_bar else range
        params = list(network.parameters())

        for idx in r(num_samples + reject):
            if idx >= reject:
                states.append(copy.deepcopy(network.state_dict()))

            accept = self.step(params, negative_log_prob)
            if accept:
                num_accept += 1

        print(f'Acceptance ratio: {num_accept / (num_samples + reject)}')
        return states

    @torch.no_grad()
    def step(self, params, negative_log_prob):
        """
        Performs a single forward evolution of Hamiltonian Monte Carlo.
        This should be repeated numerous times to get multiple parameter samples
        from multiple energy levels.

        negative_log_prob should be defined through the function of study as we traverse
        the parameters space directly on this network (using pass by reference).
        Therefore, we assume it is a parameterless function.
        """

        # Collect q0 - note that this is just a reference copy;
        # not a deep copy (important to leapfrog and acceptance step).
        q0 = TensorList(params)

        # Autograd magic
        if isinstance(negative_log_prob, NegativeLogProb):
            dVdq = negative_log_prob.dVdq
        else:
            @torch.enable_grad()
            def dVdq():
                output = negative_log_prob()

                return torch.autograd.grad(output, q0)

        # Sample initial momentum
        momentum_dist = distributions.Normal(0, 1)
        p0 = TensorList([momentum_dist.sample(param.size()) for param in q0])

        # Compute initial energy before we start changing parameters
        start_log_p = negative_log_prob() + self.kinetic_energy(p0)

        # Save initial weights to recover if we failed to accept
        q_start = copy.deepcopy(q0)

        # Integrate over our path to get a new position and momentum
        q_new, p_new = self.leapfrog(q0, p0, dVdq)

        # Check Metropolis acceptance criterion
        new_log_p = negative_log_prob() + self.kinetic_energy(p_new)

        acceptance_probability = np.exp(start_log_p - new_log_p)
        accept = np.random.rand() < acceptance_probability

        if not accept:
            # Reject - thus overwrite current (new) weights with previous values
            for q0_tensor, q_start_tensor in zip(q0, q_start):
                q0_tensor.copy_(q_start_tensor)

        return accept

    def kinetic_energy(self, p):
        if self.kinetic_energy_type == 'log_prob_sum':
            momentum_dist = distributions.Normal(0, 1)
            return -torch.stack([momentum_dist.log_prob(param).sum() for param in p]).sum()
        elif self.kinetic_energy_type == 'dot_product':
            return torch.stack([(param ** 2).sum() for param in p]).sum() / 2
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def leapfrog(self, q: TensorList, p: TensorList, dVdq):
        # We don't take a step backwards, but just move the negation from dVdq because
        # negation of a list is rarely a good idea.
        p -= TensorList(dVdq()) * (self.step_size / 2)

        for _ in range(self.num_steps - 1):
            q += p * self.step_size
            p -= TensorList(dVdq()) * self.step_size

        q += p * self.step_size
        p -= TensorList(dVdq()) * (self.step_size / 2)

        return q, -p


class StochasticGradientHMC:
    def __init__(self, step_size=0.0005, num_steps=100, momentum_decay=0.05):
        self.step_size = step_size
        self.num_steps = num_steps
        self.momentum_decay = momentum_decay
        self.kinetic_energy_type = 'log_prob_sum'  # Choices: log_prob_sum, dot_product

    def sample(self, network, negative_log_prob, num_samples=1000, reject=0, progress_bar=True):
        states = []

        r = trange if progress_bar else range
        params = list(network.parameters())

        for idx in r(num_samples + reject):
            if idx >= reject:
                states.append(copy.deepcopy(network.state_dict()))

            self.step(params, negative_log_prob)

        return states

    @torch.no_grad()
    def step(self, params, negative_log_prob):
        """
        Performs a single forward evolution of SG Hamiltonian Monte Carlo.
        This should be repeated numerous times to get multiple parameter samples
        from multiple energy levels.

        negative_log_prob should be defined through the function of study as we traverse
        the parameters space directly on this network (using pass by reference).
        Therefore, we assume it is a parameterless function.
        """

        # Collect q - note that this is just a reference copy;
        # not a deep copy (important to leapfrog and acceptance step).
        q = TensorList(params)

        # Autograd magic
        if isinstance(negative_log_prob, NegativeLogProb):
            dVdq = negative_log_prob.dVdq
        else:
            @torch.enable_grad()
            def dVdq():
                output = negative_log_prob()

                return torch.autograd.grad(output, q)

        # Sample initial momentum
        momentum_dist = distributions.Normal(0, 1)
        p = TensorList([momentum_dist.sample(param.size()) for param in q])

        # Integrate over our path to get a new position and momentum
        self.propose(q, p, dVdq)

    @torch.no_grad()
    def propose(self, q: TensorList, p: TensorList, dVdq):
        for _ in range(self.num_steps):
            q += p * self.step_size

            p -= self.step_size * self.momentum_decay
            p -= TensorList(dVdq()) * self.step_size
            sigma = torch.clamp(torch.tensor(2 * self.momentum_decay * self.step_size), min=1e-16)
            p += torch.normal(mean=0., std=sigma)
