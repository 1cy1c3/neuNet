from com.runekrauss.neunet.network import Network


def main():
    """
    A single layer perceptron fails, for example, because of the XOR problem. A multi layer perceptron is a deep
    neuronal network. It has at least three layers. Here, there are 4*3 + 3*1 = 15 connections.
    """
    network = Network()
    input_neuron1 = network.create_input_neuron()
    input_neuron2 = network.create_input_neuron()
    input_neuron3 = network.create_input_neuron()
    input_neuron4 = network.create_input_neuron()
    network.create_hidden_neurons(3)
    output_neuron1 = network.create_output_neuron()
    weights = [
        10, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0,
        10, 0, 0
    ]
    network.create_full_mesh(weights)
    input_neuron1.value = 1
    input_neuron2.value = 2
    input_neuron3.value = 3
    input_neuron4.value = 4
    print(output_neuron1.value)


if __name__ == '__main__':
    main()
