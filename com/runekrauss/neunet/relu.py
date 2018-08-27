from com.runekrauss.neunet.activation import Activation

class Relu(Activation):
    """
    This function is easy to calculate and there is a more effective training compared to sigmoid. However, it can
    explode and is not differentiated into 0.
    """

    def activate(self, x):
        """
        Stands for max(x) = {0 for x <= 0, x otherwise}.
        """
        return max(0, x)

    def derivate(self, x):
        """
        The derivation of the ReLU function for negative values is 0 and for all positive values 1.
        """
        return 0 if x < 0 else 1
