import copy
import math

import numpy as np
import torch
from torch import optim, distributions
from tqdm import trange

from bayne.nll import NegativeLogProb
from bayne.util import TensorList


class dVdqMixin:
    def grad(self, q0, negative_log_prob):
        if isinstance(negative_log_prob, NegativeLogProb):
            dVdq = negative_log_prob.dVdq
        else:
            @torch.enable_grad()
            def dVdq():
                output = negative_log_prob()

                return torch.autograd.grad(output, q0)

        return dVdq


class HamiltonianMonteCarlo(dVdqMixin):
    def __init__(self, step_size=1e-4, num_steps=50):
        self.step_size = step_size
        self.num_steps = num_steps
        self.kinetic_energy_type = 'log_prob_sum'  # Choices: log_prob_sum, dot_product

    def sample(self, mcmc, negative_log_prob, num_samples=1000, reject=0, progress_bar=True):
        num_accept = 0

        r = trange if progress_bar else range

        # Collect q0 - note that this is just a reference copy;
        # not a deep copy (important to leapfrog and acceptance step).
        params = TensorList(mcmc.parameters())

        for idx in r(num_samples + reject):
            if idx >= reject:
                mcmc.save()

            accept = self.step(params, negative_log_prob)
            if accept:
                num_accept += 1

        print(f'Acceptance ratio: {num_accept / (num_samples + reject)}')

    @torch.no_grad()
    def step(self, q0, negative_log_prob):
        """
        Performs a single forward evolution of Hamiltonian Monte Carlo.
        This should be repeated numerous times to get multiple parameter samples
        from multiple energy levels.

        negative_log_prob should be defined through the function of study as we traverse
        the parameters space directly on this network (using pass by reference).
        Therefore, we assume it is a parameterless function.
        """
        # Autograd magic
        dVdq = self.grad(q0, negative_log_prob)

        # Sample initial momentum
        momentum_dist = distributions.Normal(0, 1)
        p0 = TensorList([momentum_dist.sample(param.size()).to(param.device) for param in q0])

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


class StochasticGradientHMC(dVdqMixin):
    def __init__(self, step_size=1e-4, num_steps=50, momentum_decay=0.05, grad_noise=0.01):
        self.step_size = step_size
        self.num_steps = num_steps
        self.momentum_decay = momentum_decay
        self.grad_noise = grad_noise
        assert momentum_decay >= grad_noise

    def sample(self, mcmc, negative_log_prob, num_samples=1000, reject=0, progress_bar=True):
        r = trange if progress_bar else range

        # Collect q - note that this is just a reference copy;
        # not a deep copy (important to leapfrog and acceptance step).
        params = TensorList(mcmc.parameters())

        # Autograd magic
        dVdq = self.grad(params, negative_log_prob)

        for idx in r(num_samples + reject):
            if idx >= reject:
                mcmc.save()

            self.step(params, dVdq)

    @torch.no_grad()
    def step(self, q, dVdq):
        """
        Performs a single forward evolution of SG Hamiltonian Monte Carlo.
        This should be repeated numerous times to get multiple parameter samples
        from multiple energy levels.

        negative_log_prob should be defined through the function of study as we traverse
        the parameters space directly on this network (using pass by reference).
        Therefore, we assume it is a parameterless function.
        """

        # Sample initial momentum
        momentum_dist = distributions.Normal(0, math.sqrt(self.step_size))
        v = TensorList([momentum_dist.sample(param.size()).to(param.device) for param in q])

        for step in range(self.num_steps):
            q += v

            v *= 1 - self.momentum_decay
            v -= TensorList(dVdq()) * self.step_size
            sigma = torch.sqrt(torch.tensor(2 * (self.momentum_decay - self.grad_noise) * self.step_size))
            dist = distributions.Normal(0, sigma)
            samples = TensorList([dist.sample(x.size()).to(x.device) for x in v])
            v += samples


class CyclicalStochasticGradientHMC(dVdqMixin):
    def __init__(self, num_cycles=4, initial_step_size=1e-6, num_steps=50, momentum_decay=0.05, grad_noise=0.01, reset_after_cycle=False):
        self.num_cycles = num_cycles
        self.initial_step_size = initial_step_size
        self.num_steps = num_steps
        self.momentum_decay = momentum_decay
        self.grad_noise = grad_noise
        assert momentum_decay >= grad_noise
        self.reset_after_cycle = reset_after_cycle

    def sample(self, mcmc, negative_log_prob, num_samples=20, reject=200, progress_bar=True, inner_progress_bar=False):
        assert num_samples % self.num_cycles == 0 and reject % self.num_cycles == 0,\
            f'Number of samples ({num_samples}) and rejects ({reject}) should be a multiple of the number of cycles ({self.num_cycles})'
        iterations_per_cycle = int((num_samples + reject) // self.num_cycles)
        steps_per_cycle = iterations_per_cycle * self.num_steps
        exploration_steps_per_cycle = int(reject // self.num_cycles)

        # Collect q - note that this is just a reference copy;
        # not a deep copy (important to leapfrog and acceptance step).
        params = TensorList(mcmc.parameters())

        # Autograd magic
        dVdq = self.grad(params, negative_log_prob)

        r = trange(self.num_cycles, desc='Cycle') if progress_bar else range(self.num_cycles)
        for cycle in r:
            if self.reset_after_cycle:
                mcmc.reset()

            cr = trange(iterations_per_cycle, desc='Cycle iteration') if inner_progress_bar else range(iterations_per_cycle)

            for it in cr:
                if it == 0:
                    print('Starting exploration')
                elif it == exploration_steps_per_cycle:
                    print('Starting sampling')

                exploration = it < exploration_steps_per_cycle

                if exploration:
                    self.step_exploration(params, dVdq, it, steps_per_cycle)
                else:
                    self.step_sampling(params, dVdq, it, steps_per_cycle)
                    mcmc.save()

    def step_size(self, it, step, steps_per_cycle):
        step = it * self.num_steps + step
        iteration_percentage = step / float(steps_per_cycle)
        return self.initial_step_size / 2 * (math.cos(math.pi * iteration_percentage) + 1)

    @torch.no_grad()
    def step_exploration(self, q, dVdq, it, steps_per_cycle):
        for step in range(self.num_steps):
            step_size = self.step_size(it, step, steps_per_cycle)

            q -= TensorList(dVdq()) * step_size

    @torch.no_grad()
    def step_sampling(self, q, dVdq, it, steps_per_cycle):
        """
        Performs a single forward evolution of SG Hamiltonian Monte Carlo.
        This should be repeated numerous times to get multiple parameter samples
        from multiple energy levels.

        negative_log_prob should be defined through the function of study as we traverse
        the parameters space directly on this network (using pass by reference).
        Therefore, we assume it is a parameterless function.
        """
        v = TensorList([torch.zeros_like(param) for param in q])
        samples = TensorList([torch.zeros_like(param) for param in q])

        # Sample initial momentum
        step_size = self.step_size(it, 0, steps_per_cycle)
        v.normal_(0, math.sqrt(step_size))

        for step in range(self.num_steps):
            step_size = self.step_size(it, step, steps_per_cycle)
            q += v

            v *= 1 - self.momentum_decay
            v -= TensorList(dVdq()) * step_size
            sigma = math.sqrt(2 * (self.momentum_decay - self.grad_noise) * step_size)
            samples.normal_(0, sigma)
            v += samples


class CyclicalStochasticGradientLD(dVdqMixin):
    def __init__(self, num_cycles=4, initial_step_size=1e-6, num_steps=50, reset_after_cycle=False):
        self.num_cycles = num_cycles
        self.initial_step_size = initial_step_size
        self.num_steps = num_steps
        self.reset_after_cycle = reset_after_cycle

    def sample(self, mcmc, negative_log_prob, num_samples=20, reject=200, progress_bar=True, inner_progress_bar=False):
        assert num_samples % self.num_cycles == 0 and reject % self.num_cycles == 0,\
            f'Number of samples ({num_samples}) and rejects ({reject}) should be a multiple of the number of cycles ({self.num_cycles})'
        iterations_per_cycle = int((num_samples + reject) // self.num_cycles)
        steps_per_cycle = iterations_per_cycle * self.num_steps
        exploration_steps_per_cycle = int(reject // self.num_cycles)

        # Collect q - note that this is just a reference copy;
        # not a deep copy (important to leapfrog and acceptance step).
        params = TensorList(mcmc.parameters())

        # Autograd magic
        dVdq = self.grad(params, negative_log_prob)

        r = trange(self.num_cycles, desc='Cycle') if progress_bar else range(self.num_cycles)
        for cycle in r:
            if self.reset_after_cycle:
                mcmc.reset()

            cr = trange(iterations_per_cycle, desc='Cycle iteration') if inner_progress_bar else range(iterations_per_cycle)

            for it in cr:
                if it == 0:
                    print('Starting exploration')
                elif it == exploration_steps_per_cycle:
                    print('Starting sampling')

                exploration = it < exploration_steps_per_cycle

                if exploration:
                    self.step_exploration(params, dVdq, it, steps_per_cycle)
                else:
                    self.step_sampling(params, dVdq, it, steps_per_cycle)
                    mcmc.save()

    def step_size(self, it, step, steps_per_cycle):
        step = it * self.num_steps + step
        iteration_percentage = step / float(steps_per_cycle)
        return self.initial_step_size / 2 * (math.cos(math.pi * iteration_percentage) + 1)

    @torch.no_grad()
    def step_exploration(self, q, dVdq, it, steps_per_cycle):
        for step in range(self.num_steps):
            step_size = self.step_size(it, step, steps_per_cycle)

            q -= TensorList(dVdq()) * step_size

    @torch.no_grad()
    def step_sampling(self, q, dVdq, it, steps_per_cycle):
        """
        Performs a single forward evolution of SG Langevin Dynamics.
        This should be repeated numerous times to get multiple parameter samples
        from multiple energy levels.

        negative_log_prob should be defined through the function of study as we traverse
        the parameters space directly on this network (using pass by reference).
        Therefore, we assume it is a parameterless function.
        """
        for step in range(self.num_steps):
            step_size = self.step_size(it, step, steps_per_cycle)

            q -= TensorList(dVdq()) * step_size
            sigma = math.sqrt(2 * step_size)
            dist = distributions.Normal(0, sigma)
            samples = TensorList([dist.sample(x.size()).to(x.device) for x in q])
            q += samples