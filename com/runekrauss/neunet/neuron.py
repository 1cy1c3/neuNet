from abc import ABCMeta, abstractmethod

class Neuron:
    """
    Describes the neuron that is connected via synapses.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __get_value(self):
        raise NotImplementedError('subclasses must override get_value()!')
