from com.runekrauss.neunet.neuron import Neuron
from com.runekrauss.neunet.identity import Identity
from com.runekrauss.neunet.hyperbolic_tangent import HyperbolicTangent
from com.runekrauss.neunet.piecewise import Piecewise
from com.runekrauss.neunet.sigmoid import Sigmoid
from com.runekrauss.neunet.relu import Relu

class WorkingNeuron(Neuron):
    """
    In this case, the working neuron represents an output neuron or an hidden neuron. With an output neuron you only
    get access to the result. With the help of a hidden neuron, functions can be reconstructed. For example, the digest
    data set can also be learned from pixels.
    """

    __activation = Relu()

    def __init__(self):
        """
        The working neuron has mostly more than one synapse. The result is the sum of synapse values.
        """
        self.__synapses = []

    def __get_value(self):
        """
        The input of a neuron is equal to the sum of the multiplication of the previous weights and inputs. Then, this
        value is modified again.

        :return: Activation level
        """
        input_sum = sum(synapse.value for synapse in self.__synapses)
        return self.__activation.activate(input_sum)

    def add_synapse(self, synapse):
        self.__synapses.append(synapse)

    def delta_learning(self, epsilon, small_delta):
        """
        Computes the big delta regarding \Delta w_{ik} = \epsilon * \delta_i * a_k whereby \epsilon stands for the
        learning factor, a_k means the activation level and \delta_i is the difference between should(a_k) and
        is(a_k).

        :param epsilon: Learning factor between 0 and 1
        :param small_delta: Difference between actual and nominal values
        """
        big_delta_factor = self.__activation.derivate(self.value) * epsilon * small_delta
        for synapse in self.__synapses:
            big_delta = big_delta_factor * synapse.neuron.value
            synapse.add_weight(big_delta)

    value = property(__get_value)
