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
        # When adjusting the weight, note the previous weight change (one-way fitting)
        self.__momentum = 0
        self.__weight_add = 0

    def __get_value(self):
        return self.__neuron.value * self.__weight

    def __get_neuron(self):
        return self.__neuron

    def __get_weight(self):
        return self.__weight

    def apply_batch(self):
        """
        Applies batch learning, i. e. the weight is only adjusted after several images. To avoid the problem with
        plateaus and oscillations, the momentum is used. Batch learning supports this procedure regarding the
        performance.
        """
        self.__momentum += self.__weight_add
        self.__momentum *= 0.9
        self.__weight += self.__weight_add + self.__momentum
        self.__weight_add = 0

    def add_weight(self, weight_delta):
        """
        Updates the specific synapse with the calculated big delta.

        :param weight_delta: Big delta
        """
        self.__weight_add += weight_delta

    value = property(__get_value)
    neuron = property(__get_neuron)
    weight = property(__get_weight)
