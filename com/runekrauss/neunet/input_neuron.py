from com.runekrauss.neunet.neuron import Neuron

class InputNeuron(Neuron):
    """
    Get input values and forwards it. The default value is 0. For this reason, a direct access is possible.
    """
    def __init__(self):
        self.__value = 0;

    def __get_value(self):
        return self.__value

    def __set_value(self, value):
        self.__value = value

    value = property(__get_value, __set_value)
