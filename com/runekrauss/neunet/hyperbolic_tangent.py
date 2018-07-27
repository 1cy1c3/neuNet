from com.runekrauss.neunet.activation import Activation
import math

class HyperbolicTangent(Activation):
    """
    The hyperbolic tangent is the solution to the differential equation f' = 1 - f^2 with f(0) = 0.
    """

    def activate(self, input):
        """
        Stands for activate(input) = sinh(input) / cosh(input).
        """
        return math.sinh(input) / math.cosh(input)