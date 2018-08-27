from com.runekrauss.neunet.activation import Activation
import math


class HyperbolicTangent(Activation):
    """
    The hyperbolic tangent is the solution to the differential equation f' = 1 - f^2 with f(0) = 0.
    """

    def activate(self, x):
        """
        Stands for activate(x) = sinh(x) / cosh(x).
        """
        return math.sinh(x) / math.cosh(x)

    def derivate(self, x):
        """
        Stands for activate'(x) = 1 - activate(x)^2
        """
        return 1 - self.activate(x)**2
