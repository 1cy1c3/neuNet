from com.runekrauss.neunet.network import Network

def main():
    """
    A single layer perceptron has two layers for input and output neurons. So it is a linear classifier for two classes
    with f(x) = {1 if wx + b > 0, 0 otherwise}. The bias reduces the learning problem to w and and shifts the threshold
    value. In addition, it is also possible to learn learning curves that not only go through the origin (for example
    or, and, ...). Here, there are 4 * 1 = 4 connections.
    """
    network = Network()
    input_neuron1 = network.create_input_neuron()
    input_neuron2 = network.create_input_neuron()
    input_neuron3 = network.create_input_neuron()
    input_neuron4 = network.create_input_neuron()
    output_neuron1 = network.create_output_neuron()
    weights = [-10, 0, 0, 0]
    network.create_full_mesh(weights)
    input_neuron1.value = 1
    input_neuron2.value = 2
    input_neuron3.value = 3
    input_neuron4.value = 4
    print(output_neuron1.value)

if __name__ == '__main__':
    main()