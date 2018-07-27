from com.runekrauss.neunet.input_neuron import InputNeuron
from com.runekrauss.neunet.synapse import Synapse
from com.runekrauss.neunet.working_neuron import WorkingNeuron

class Network:
    """
    A neural network is a mathematical replica of the neurons in our brain, but neural network != brain because
    biological neurons are also time-sensitive. It is possible to create single layer perceptrons and multi layer
    perceptrons.
    """
    def __init__(self):
        """
        Creates empty input, hidden and output neurons.
        """
        self.__input_neurons = []
        self.__hidden_neurons = []
        self.__output_neurons = []

    def create_input_neuron(self):
        input_neuron = InputNeuron()
        self.__input_neurons.append(input_neuron)
        return input_neuron

    def create_hidden_neurons(self, amount):
        if amount <= 0:
            raise Exception("The amount is illegal!")
        for i in range(0, amount):
            self.__hidden_neurons.append(WorkingNeuron())

    def create_output_neuron(self):
        output_neuron = WorkingNeuron()
        self.__output_neurons.append(output_neuron)
        return output_neuron

    def delta_learning(self, nominal_values, epsilon):
        """
        The delta rule is a gradient descent learning rule for updating the weights of the inputs to artificial neurons
        in a single-layer neural network. First, initializes the weight vector with random numbers or zeros. Afterwards,
        computes an output for each sample and updates the weights. This process is repeated until the error is smaller
        than the threshold value. Overall, the error function is to be minimized.

        :param nominal_values: Nominal values regarding supervised learning
        :param epsilon: Learning factor between 0 and 1
        """
        if len(nominal_values) != len(self.__output_neurons):
            raise Exception("The length of shoulds is illegal!")
        if len(self.__hidden_neurons) != 0:
            raise Exception("The are hidden neurons!")
        i = 0
        while i < len(nominal_values):
           small_delta = nominal_values[i] - self.__output_neurons[i].value
           # Update weight
           self.__output_neurons[i].delta_learning(epsilon, small_delta)
           i = i + 1

    def create_full_mesh(self, weights):
        """
        Creates a mesh, it means a graph where all neurons are connected. Supports single layer perceptrons and multi
        layer perceptrons.

        :param weights: Weights as a multiplicative factor for synapses
        """
        if len(weights) == 0:
            # Single layer perceptron
            if len(self.__hidden_neurons) == 0:
                for output_neuron in self.__output_neurons:
                    for input_neuron in self.__input_neurons:
                        output_neuron.add_synapse(Synapse(input_neuron, 0))
            # Multi layer perceptron
            else:
                # Connect output neurons with hidden neurons
                for output_neuron in self.__output_neurons:
                    for hidden_neuron in self.__hidden_neurons:
                        output_neuron.add_synapse(Synapse(hidden_neuron, 0))
                # Connect hidden neurons with input neurons
                for hidden_neuron in self.__hidden_neurons:
                    for input_neuron in self.__input_neurons:
                        hidden_neuron.add_synapse(Synapse(input_neuron, 0))
        else:
            if len(self.__hidden_neurons) == 0:
                if len(weights) != len(self.__input_neurons) * len(self.__output_neurons):
                    raise Exception("The weights size is illegal!")
                index = 0
                for output_neuron in self.__output_neurons:
                    for input_neuron in self.__input_neurons:
                        output_neuron.add_synapse(Synapse(input_neuron, weights[index]))
                        index += 1
            else:
                if len(weights) != len(self.__input_neurons) * len(self.__hidden_neurons) + len(self.__hidden_neurons) * len(self.__output_neurons):
                    raise Exception("The weights size is illegal!")
                index = 0
                for hidden_neuron in self.__hidden_neurons:
                    for input_neuron in self.__input_neurons:
                        hidden_neuron.add_synapse(Synapse(input_neuron, weights[index]))
                        index += 1
                for output_neuron in self.__output_neurons:
                    for hidden_neuron in self.__hidden_neurons:
                        output_neuron.add_synapse(Synapse(hidden_neuron, weights[index]))
                        index += 1
