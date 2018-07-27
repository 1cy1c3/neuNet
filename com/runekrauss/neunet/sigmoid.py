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
        return 1.0 / (1.0 + math.pow(math.e, -input))