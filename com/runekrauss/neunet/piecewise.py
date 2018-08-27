from com.runekrauss.neunet.activation import Activation

class Piecewise(Activation):
    """
    A piecewise function is a function that is defined on a sequence of intervals.
    """

    def activate(self, x):
        """
        Stands for activate(x) = {0 for x < 0, 1 for x >= 0}.
        """
        return 0 if x < 0 else 1

    def derivate(self, x):
        """
        Is not the right derivation, but still works well.
        """
        return 1
