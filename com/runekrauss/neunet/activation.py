from abc import ABCMeta, abstractmethod

class Activation:
    """
    Stands for the activation function to fire a neuron. The functions are not linear, so that hidden layers also have
    a function at all. Overall, the pulse is modified. If a threshold value is exceeded, the neuron is fired
    (activation level).
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def activate(self, input):
        """
        :param input: Input for computing the image of the function
        :return: Value of the respective function
        """
        raise NotImplementedError('subclasses must override activate()!')