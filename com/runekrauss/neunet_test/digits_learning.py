from com.runekrauss.neunet.network import Network
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random

# Load digits dataset (contains 1797 images or digits whereby each image consisting of 8*8 pixels
digits = load_digits()

# Pixels and digits
data, target = digits.data, digits.target

# Cross validation: Divide the data into complementary subsets
train_data, test_data, train_target, test_target = train_test_split(data, target,
                                                    test_size=0.1,
                                                    random_state=1)

# Create 8*8 input neurons
input_neurons = [[0] * 8] * 8

# Create 10 output neurons
output_neurons = [0] * 10

# Number of hidden neurons
number_of_hidden_neurons = 100

# Initialize the neuronal network
network = Network()

class ProbabilityDigit:
    """
    Holds the propabilities for every digit (0-9).
    """
    def __init__(self, digit, probability):
        self.digit = digit
        self.probability = probability
    def __lt__(self, other):
        return self.probability < other.probability
    def __repr__(self):
        return "%s %s" % (self.digit, self.probability)


def test():
    """
    Represents the test stage to evaluate how good the learning process is (the accuracy is checked). It takes the
    current weights regarding the learning stage.
    """
    correct = 0
    incorrect = 0
    row = 0
    while row < len(test_data):
        col = 0
        network.reset()
        # Assign sample to the input neurons
        for x in range(len(input_neurons)):
            for y in range(len(input_neurons[x])):
                # Adapt value between 0/1
                input_neurons[x][y].value = test_data[row][col] / 16
                col = col + 1
        # Outputs (hyperplanes) for discrete probabilities regarding the softmax layer
        probs = [0] * 10
        for k in range(0, 10):
            probs[k] = ProbabilityDigit(k, output_neurons[k].value)
        # Sort them by the highest probability
        probs.sort(reverse=True)
        was_correct = False
        # Compare the digit with the highest propability with the right digit of the test set
        for i in range(1):
            if test_target[row] == probs[i].digit:
                was_correct = True
        if was_correct:
            correct = correct + 1
        else:
            incorrect = incorrect + 1
        row = row + 1
    # Calculate the accuracy and print it
    accuracy = correct / (correct + incorrect)
    print(accuracy)

def familiarize_with_digits():
    """
    Familiarizes himself with the digits dataset.
    """
    # Create feature matrix
    data = digits.data
    # Create target vector
    target = digits.target
    # size of feature matrix
    print(data.shape)
    # size of target vector
    print(target.shape)
    # View the first observation's feature values
    print(data[0])
    # Visualize the first observation's feature values as an image
    plt.gray()
    plt.matshow(digits.images[2])
    plt.show()

def main():
    """
    Trains and tests (with other data) the digits dataset with the delta learning:
    1. Creates the neuronal network with 8*8 input and 10 output neurons.
    2. Fills the 8*8*10 weight vector with random data.
    3. Connects the neurons to each other.
    4. while 1:
        Tests the neuronal network with the current weights.
            Fills the input neurons with the pixel values of the digit.
            Gets the outputs as propabilities for the digits.
            Sorts them by the highest probability and compare it with the right label.
            Calculates the accuracy.
        Fills the input neurons with the pixel values of the digit.
        Sets the respective digit to 1.
        Uses delta learning to train the neuronal network.
    """
    # Create the neuronal network with 8*8 input (matrix) and 10 output neurons as well as 100 hidden neurons
    for i in range(len(input_neurons)):
        for j in range(len(input_neurons[i])):
            input_neurons[i][j] = network.create_input_neuron()
    for i in range(len(output_neurons)):
        output_neurons[i] = network.create_output_neuron()
    number_of_hidden_neurons = 100
    network.create_hidden_neurons(number_of_hidden_neurons)

    # Fill the 8*8*10 weight vector with random data and connect the neurons to each other
    weights = []
    synapses = 8*8*number_of_hidden_neurons + number_of_hidden_neurons*10
    for i in range(synapses):
        weights.append(random.random())
    network.create_full_mesh(weights)

    # Learning stage
    epsilon = 0.005
    while 1:
        test()
        row = 0
        while row < len(train_data):
            col = 0
            # Assign sample to the input neurons
            for x in range(len(input_neurons)):
                for y in range(len(input_neurons[x])):
                    # Adapt value between 0/1
                    input_neurons[x][y].value = train_data[row][col] / 16
                    col = col + 1
            # Use one hot encoding (labels are nominally scaled) to ensure equal distance regarding the digits
            shoulds = [0] * 10
            shoulds[train_target[row]] = 1
            # Delta learning
            network.backpropagation(shoulds, epsilon)
            row = row + 1
        epsilon *= 0.9

if __name__ == '__main__':
    main()
