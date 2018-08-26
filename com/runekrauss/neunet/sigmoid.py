from com.runekrauss.neunet.activation import Activation
import math

class Sigmoid(Activation):
    """
    A sigmoid function is a mathematical function having a characteristic "S"-shaped curve. Furthermore, this function
    is continuous. A small change changes the overall result only minimally.
    """

    def activate(self, input):
        """
        Stands for activate(input) = 1 / (1+e^{-input}).
        """
        if input < 0:
            return 1 - 1 / (1 + math.exp(input))
        return 1 / (1 + math.exp(-input))

    def derivate(self, input):
        """
        Stands for activate'(input) = activate(input) * ( 1 - activate(input) )
        """
        sigmoid = self.activate(input)
        return sigmoid * (1 - sigmoid)