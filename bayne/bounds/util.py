import types
from contextlib import contextmanager


@contextmanager
def optional(condition, context_manager):
    if condition:
        with context_manager:
            yield
    else:
        yield


@contextmanager
def notnull(context_manager):
    if context_manager is not None:
        with context_manager:
            yield
    else:
        yield


def add_method(class_or_obj, name, func):
    if not isinstance(class_or_obj, type):
        func = types.MethodType(func, class_or_obj)

    setattr(class_or_obj, name, func)
