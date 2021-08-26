import warnings
from collections import deque, abc
from typing import Optional, Iterable, Iterator

import torch
from torch import nn, Tensor


class ParameterQueue(nn.Module):
    r"""Holds parameters in a queue.

    :class:`~bayne.container.ParameterQueue`

    Args:
        cur: Current value of the parameter of :class:`~torch.Tensor`
        parameters (iterable, optional): an iterable of :class:`~torch.Tensor` to add
        maxlen (optional): Bound the length of the queue
    """

    def __init__(self, cur: Tensor, parameters: Optional[Iterable[Tensor]] = None, maxlen: Optional[int] = None) -> None:
        super(ParameterQueue, self).__init__()
        self.deque = []
        self.maxlen = maxlen
        self.active_index = None

        self.cur = nn.Parameter(cur)

        if parameters is not None:
            self.extend(parameters)

    def __len__(self) -> int:
        return len(self.deque)

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self.deque)

    def __iadd__(self, parameters: Iterable[Tensor]) -> 'ParameterQueue':
        return self.extend(parameters)

    def append(self, parameter: Tensor) -> 'ParameterQueue':
        """Appends a given tensors at the end of the list.

        Args:
            parameter (Tensor): parameter to append
        """
        self.deque.append(parameter)
        self.crop()
        return self

    def crop(self):
        self.deque = self.deque[-self.maxlen:]

    def extend(self, parameters: Iterable[Tensor]) -> 'ParameterQueue':
        """Appends tensors from a Python iterable to the end of the list.

        Args:
            parameters (iterable): iterable of parameters to append
        """
        if not isinstance(parameters, abc.Iterable):
            raise TypeError("ParameterQueue.extend should be called with an "
                            "iterable, but got " + type(parameters).__name__)

        self.deque.extend(parameters)
        self.crop()
        return self

    def save(self):
        self.append(self.cur.detach().clone())

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(destination, prefix, keep_vars)
        state_dict['deque'] = torch.stack(self.deque)

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        self.extend(state_dict.pop('deque').unbind())

        super().load_state_dict(state_dict, strict)

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in enumerate(self.deque):
            size_str = 'x'.join(str(size) for size in p.size())
            device_str = '' if not p.is_cuda else ' (GPU {})'.format(p.get_device())
            parastr = 'Parameter containing: [{} of size {}{}]'.format(torch.typename(p), size_str, device_str)
            child_lines.append('  (' + str(k) + '): ' + parastr)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('ParameterQueue should not be called.')

    def _replicate_for_data_parallel(self):
        warnings.warn("bayne.container.ParameterQueue is being used with DataParallel but this is not "
                      "supported. This list will appear empty for the models replicated "
                      "on each GPU except the original one.")

        return super(ParameterQueue, self)._replicate_for_data_parallel()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        def convert_arg(arg):
            if not isinstance(arg, ParameterQueue):
                return arg

            if arg.active_index is None:
                return arg.cur

            if 0 <= arg.active_index < len(self):
                return arg.deque[arg.active_index]

            raise ValueError('Active index out of bounds')

        args = list(map(convert_arg, args))
        return func(*args, **kwargs)
