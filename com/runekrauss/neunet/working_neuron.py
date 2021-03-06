from com.runekrauss.neunet.neuron import Neuron
from com.runekrauss.neunet.sigmoid import Sigmoid

class WorkingNeuron(Neuron):
    """
    In this case, the working neuron represents an output neuron or an hidden neuron. With an output neuron you only
    get access to the result. With the help of a hidden neuron, functions can be reconstructed. For example, the digest
    data set can also be learned from pixels.
    """

    __activation = Sigmoid()

    def __init__(self):
        """
        The working neuron has mostly more than one synapse. The result is the sum of synapse values.
        """
        self.__synapses = []
        # Difference between actual and nominal values
        self.__small_delta = 0.0
        # Actual value
        self.__value = 0.0
        # Has the value already been calculated?
        self.__is_calculated = False

    def __get_value(self):
        """
        The input of a neuron is equal to the sum of the multiplication of the previous weights and inputs. Then, this
        value is modified again.

        :return: Activation level
        """
        if self.__is_calculated is False:
            input_sum = sum(synapse.value for synapse in self.__synapses)
            self.__value = self.__activation.activate(input_sum)
            self.__is_calculated = True
        return self.__value

    def __get_activation(self):
        return self.__activation

    def __set_activation(self, activation):
        self.__activation = activation

    def reset(self):
        """
        Sets the small deltas for backpropagation to 0 at the beginning.
        """
        self.__is_calculated = False
        self.__small_delta = 0.0

    def add_synapse(self, synapse):
        self.__synapses.append(synapse)

    def delta_learning(self, epsilon):
        """
        Calculates the big delta regarding \Delta w_{ik} = \epsilon * \delta_i * a_k whereby \epsilon stands for the
        learning factor, a_k means the activation level and \delta_i is the difference between should(a_k) and
        is(a_k).

        :param epsilon: Learning factor between 0 and 1
        """
        big_delta_factor = self.__activation.derivate(self.value) * epsilon * self.__small_delta
        for synapse in self.__synapses:
            big_delta = big_delta_factor * synapse.neuron.value
            synapse.add_weight(big_delta)

    def calculate_output_delta(self, nominal_value):
        """
        If the neuron is an output neuron, then calculates a(nominal_value) - a(actual_value) as small delta.

        :param nominal_value: Expected should value in terms of supervised learning
        """
        self.__small_delta = nominal_value - self.value

    def backpropagate_small_delta(self):
        """
        Propagate the small delta backwards, that means the calculation of \sum_L (\delta_l * w_{li}).
        """
        for synapse in self.__synapses:
            neuron = synapse.neuron
            if isinstance(neuron, WorkingNeuron):
                neuron.__small_delta += self.__small_delta * synapse.weight

    def apply_batch_training(self):
        """
        Applies batch training regarding each connection.
        """
        for synapse in self.__synapses:
            synapse.apply_batch_training()

    value = property(__get_value)
    activation = property(__get_activation, __set_activation)
