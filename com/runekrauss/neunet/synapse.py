class Synapse:
    """
    Connects neurons with each other. The connection has a weight that is a multiplicative factor.
    """
    def __init__(self, neuron, weight):
        """
        Initializes a synapse with a neuron and weight.

        :param neuron: Neuron
        :param weight: Multiplicative factor
        """
        self.__neuron = neuron
        self.__weight = weight

    def __get_value(self):
        return self.__neuron.value * self.__weight

    def __get_neuron(self):
        return self.__neuron

    def add_weight(self, weight_delta):
        """
        Updates the specific synapse with the computed big delta.

        :param weight_delta: Big delta
        """
        self.__weight += weight_delta

    value = property(__get_value)
    neuron = property(__get_neuron)
