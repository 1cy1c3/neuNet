from com.runekrauss.neunet.activation import Activation

class Identity(Activation):
    """
    An identity function is a function that always returns the same value that was used as its argument.
    """

    def activate(self, input):
        """
        Stands for activate(input) = input.
        """
        return input

    def derivate(self, input):
        """
        It has a gradient of 1.
        """
        return 1