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
    def step(self):
        """
        Performs a single forward evolution of Hamiltonian Monte Carlo.
        This should be repeated numerous times to get multiple parameter samples
        from multiple energy levels
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

        # Integrate over our path to get a new position and momentum
        q_new, p_new = self.leapfrog(q0, p0, dVdq)

        # Check Metropolis acceptance criterion
        start_log_p = negative_log_prob(q0) - momentum_dist.log_prob(p0).sum()
        new_log_p = negative_log_prob(q_new) - momentum_dist.log_prob(p_new).sum()

        acceptance_probability = np.exp(start_log_p - new_log_p)

        if np.random.rand() < acceptance_probability:
            # Accept - thus overwrite weights with new values
            for q0_tensor, q_new_tensor in zip(q0, q_new):
                q0_tensor.copy_(q_new_tensor)

    @torch.no_grad()
    def leapfrog(self, q, p, dVdq):
        q, p = copy.deepcopy(q), copy.deepcopy(p)

        self.update_variable(p, -dVdq(q), self.step_size / 2)

        for _ in range(self.num_steps - 1):
            self.update_variable(q, p, self.step_size)
            self.update_variable(p, -dVdq(q), self.step_size)

        self.update_variable(q, p, self.step_size)
        self.update_variable(p, -dVdq(q), self.step_size / 2)

        return q, -p

    @torch.no_grad()
    def update_variable(self, x, dx, step_size):
        for x_prime, dx_prime in zip(x, dx):
            x_prime += step_size * dx_prime
