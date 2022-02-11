import types


def add_method(class_or_obj, name, func):
    if not isinstance(class_or_obj, type):
        func = types.MethodType(func, class_or_obj)

    setattr(class_or_obj, name, func)
