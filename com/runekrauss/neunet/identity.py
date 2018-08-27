from com.runekrauss.neunet.activation import Activation


class Identity(Activation):
    """
    An identity function is a function that always returns the same value that was used as its argument.
    """

    def activate(self, x):
        """
        Stands for activate(x) = x.
        """
        return x

    def derivate(self, x):
        """
        It has a gradient of 1.
        """
        return 1
