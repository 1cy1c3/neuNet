from com.runekrauss.neunet.activation import Activation

class Relu(Activation):
    """
    This function is easy to calculate and there is a more effective training compared to sigmoid. However, it can
    explode and is not differentiated into 0.
    """

    def activate(self, input):
        """
        Stands for max(input) = {0 for input <= 0, input otherwise}.
        """
        return max(0, input)

    def derivate(self, input):
        """
        The derivation of the ReLU function for negative values is 0 and for all positive values 1.
        """
        return 0 if input < 0 else 1