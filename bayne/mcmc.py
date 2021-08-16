import copy
import itertools

import torch
from torch import nn
from torch.nn import Parameter
from tqdm import trange

from bayne.distributions import PriorWeightDistribution
from bayne.sampler import HamiltonianMonteCarlo


def _non_copy_load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
    r"""Copies parameters and buffers from :attr:`state_dict` into only
    this module, but not its descendants. This is called on every submodule
    in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
    module in input :attr:`state_dict` is provided as :attr:`local_metadata`.
    For state dicts without metadata, :attr:`local_metadata` is empty.
    Subclasses can achieve class-specific backward compatible loading using
    the version number at `local_metadata.get("version", None)`.

    .. note::
        :attr:`state_dict` is not the same object as the input
        :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
        it can be modified.

    Args:
        state_dict (dict): a dict containing parameters and
            persistent buffers.
        prefix (str): the prefix for parameters and buffers used in this
            module
        local_metadata (dict): a dict containing the metadata for this module.
            See
        strict (bool): whether to strictly enforce that the keys in
            :attr:`state_dict` with :attr:`prefix` match the names of
            parameters and buffers in this module
        missing_keys (list of str): if ``strict=True``, add missing keys to
            this list
        unexpected_keys (list of str): if ``strict=True``, add unexpected
            keys to this list
        error_msgs (list of str): error messages should be added to this
            list, and will be reported together in
            :meth:`~torch.nn.Module.load_state_dict`
    """
    for hook in self._load_state_dict_pre_hooks.values():
        hook(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    persistent_buffers = {k: v for k, v in self._buffers.items() if k not in self._non_persistent_buffers_set}
    local_name_params = itertools.chain(self._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}

    parameter_names = self._parameters.keys()
    buffer_names = persistent_buffers.keys()

    for name, param in local_state.items():
        key = prefix + name
        if key in state_dict:
            input_param = state_dict[key]
            # This is used to avoid copying uninitialized parameters into
            # non-lazy modules, since they dont have the hook to do the checks
            # in such case, it will error when accessing the .shape attribute.
            is_param_lazy = torch.nn.parameter.is_lazy(param)
            # Backward compatibility: loading 1-dim tensor from 0.3.* to version 0.4+
            if not is_param_lazy and len(param.shape) == 0 and len(input_param.shape) == 1:
                input_param = input_param[0]

            if not is_param_lazy and input_param.shape != param.shape:
                # local shape should match the one in checkpoint
                error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                  'the shape in current model is {}.'
                                  .format(key, input_param.shape, param.shape))
                continue
            try:
                if name in parameter_names:
                    self._parameters[name] = Parameter(input_param.to(dtype=param.dtype, device=param.device))
                elif name in buffer_names:
                    param.copy_(input_param)
                else:
                    raise Exception('Could not find parameter {} on class'.format(name))
            except Exception as ex:
                error_msgs.append('While copying the parameter named "{}", '
                                  'whose dimensions in the model are {} and '
                                  'whose dimensions in the checkpoint are {}, '
                                  'an exception occurred : {}.'
                                  .format(key, param.size(), input_param.size(), ex.args))
        elif strict:
            missing_keys.append(key)

    if strict:
        for key in state_dict.keys():
            if key.startswith(prefix):
                input_name = key[len(prefix):]
                input_name = input_name.split('.', 1)[0]  # get the name of param/buffer/child
                if input_name not in self._modules and input_name not in local_state:
                    unexpected_keys.append(key)


def non_copy_load_state_dict_wrapper(cls):
    cls._load_from_state_dict = _non_copy_load_from_state_dict
    return cls


class MonteCarloBNN(nn.Module):
    def __init__(self, network, sampler=HamiltonianMonteCarlo(step_size=0.0005, num_steps=100)):
        super().__init__()

        self.network = network
        self.states = []
        self.sampler = sampler

    def sample(self, negative_log_prob, num_samples=1000, reject=0, progress_bar=True):
        self.states = self.sampler.sample(self.network, negative_log_prob, num_samples, reject, progress_bar)

    def forward(self, *args, state_idx=None, **kwargs):
        if state_idx is not None:
            self.network.load_state_dict(self.states[state_idx])

        return self.network(*args, **kwargs)

    def predict_dist(self, *args, num_samples=None, dim=0, **kwargs):
        preds = [self(*args, **kwargs, state_idx=idx) for idx in range(len(self.states))]
        preds = torch.stack(preds, dim=dim)
        return preds

    def predict_mean(self, *args, num_samples=None, dim=0, **kwargs):
        preds = self.predict_dist(*args, dim=dim, **kwargs)
        return preds.mean(dim=dim)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict['states'] = self.states

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.states = state_dict.pop('states')
        self.network.load_state_dict(self.states[0])

        self.apply(non_copy_load_state_dict_wrapper)

        super().load_state_dict(state_dict, strict)

    def log_prior(self, dist=PriorWeightDistribution()):
        return torch.stack([dist.log_prior(w).sum() for w in self.network.parameters()]).sum()
