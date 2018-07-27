from com.runekrauss.neunet.activation import Activation

class Relu(Activation):
    """
    This function is easy to compute and there is a more effective training compared to sigmoid. However, it can
    explode and is not differentiated into 0.
    """

    def activate(self, input):
        """
        Stands for max(input) = {0 for input <= 0, input otherwise}.
        """
        return max(0, input)