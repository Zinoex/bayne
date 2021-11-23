import abc


class Bounds(abc.ABC):
    @abc.abstractmethod
    def interval_bounds(self, model, input_bounds):
        raise NotImplementedError()

    @abc.abstractmethod
    def linear_bounds(self, model, input_bounds):
        raise NotImplementedError()
