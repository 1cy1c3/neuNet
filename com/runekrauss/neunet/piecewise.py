from com.runekrauss.neunet.activation import Activation

class Piecewise(Activation):
    """
    A piecewise function is a function that is defined on a sequence of intervals.
    """

    def activate(self, input):
        """
        Stands for activate(input) = {0 for input < 0, 1 for input >= 0}.
        """
        return 0 if input < 0 else 1

    def derivate(self, input):
        """
        Is not the right derivation, but still works well.
        """
        return 1