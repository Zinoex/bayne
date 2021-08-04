import copy

import numpy as np
import torch
from torch import optim, distributions


class HamiltonianMonteCarlo(optim.Optimizer):
    def __init__(self, params, step_size, num_steps):
        super().__init__(params, {})

        self.step_size = step_size
        self.num_steps = num_steps

    @torch.no_grad()
    def step(self, negative_log_prob):
        """
        Performs a single forward evolution of Hamiltonian Monte Carlo.
        This should be repeated numerous times to get multiple parameter samples
        from multiple energy levels.

        negative_log_prob should be defined through the function of study as we traverse
        the parameters space directly on this network (using pass by reference).
        Therefore, we assume it is a parameterless function.
        """
        # autograd magic
        dVdq = grad(negative_log_prob)

        # Collect q0 - note that this is just a reference copy;
        # not a deep copy (important to leapfrog and acceptance step).
        q0 = []
        for group in self.param_groups:
            for param in group['params']:
                q0.append(param)

        # Sample initial momentum
        momentum_dist = distributions.Normal(0, 1)
        p0 = [momentum_dist.sample(param.size()) for param in q0]

        # Compute initial energy before we start changing parameters
        start_log_p = negative_log_prob() - torch.stack([momentum_dist.log_prob(param).sum() for param in p0]).sum()

        # Save initial weights to recover if we failed to accept
        q_start = copy.deepcopy(q0)

        # Integrate over our path to get a new position and momentum
        q_new, p_new = self.leapfrog(q0, p0, dVdq)

        # Check Metropolis acceptance criterion
        new_log_p = negative_log_prob() - torch.stack([momentum_dist.log_prob(param).sum() for param in p_new]).sum()

        acceptance_probability = np.exp(start_log_p - new_log_p)

        if np.random.rand() > acceptance_probability:
            # Reject - thus overwrite current (new) weights with previous values
            for q0_tensor, q_start_tensor in zip(q0, q_start):
                q0_tensor.copy_(q_start_tensor)

    @torch.no_grad()
    def leapfrog(self, q, p, dVdq):
        # We don't take a step backwards, but just move the negation from dVdq because
        # negation of a list is rarely a good idea.
        self.update_variable(p, dVdq(q), -self.step_size / 2)

        for _ in range(self.num_steps - 1):
            self.update_variable(q, p, self.step_size)
            self.update_variable(p, dVdq(q), -self.step_size)

        self.update_variable(q, p, self.step_size)
        self.update_variable(p, dVdq(q), -self.step_size / 2)

        return q, -p

    @torch.no_grad()
    def update_variable(self, x, dx, step_size):
        for x_prime, dx_prime in zip(x, dx):
            x_prime.add_(step_size * dx_prime)
