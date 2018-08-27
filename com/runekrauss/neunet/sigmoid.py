from com.runekrauss.neunet.activation import Activation
import math


class Sigmoid(Activation):
    """
    A sigmoid function is a mathematical function having a characteristic "S"-shaped curve. Furthermore, this function
    is continuous. A small change changes the overall result only minimally.
    """

    def activate(self, x):
        """
        Stands for activate(x) = 1 / (1+e^{-x}).
        """
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))

    def derivate(self, x):
        """
        Stands for activate'(x) = activate(x) * ( 1 - activate(x) )
        """
        sigmoid = self.activate(x)
        return sigmoid * (1 - sigmoid)
