from abc import ABCMeta, abstractmethod


class Activation:
    """
    Stands for the activation function to fire a neuron. The functions are not linear, so that hidden layers also have
    a function at all. Overall, the pulse is modified. If a threshold value is exceeded, the neuron is fired
    (activation level). To make the networks more powerful, there are derivations.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def activate(self, x):
        """
        Modifies a pulse.

        :param x: Input for calculation the image of the function
        :return: Value of the respective function
        """
        raise NotImplementedError('subclasses must override activate()!')

    def derivate(self, x):
        """
        Indicates the derivation of a function.

        :param x: Input for computing the image of the function
        :return: Value of the derived function
        """
        raise NotImplementedError('subclasses must override derivative()!')
