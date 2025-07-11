import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    """
    Hyperbolic tangent activation layer.
    """

    def __init__(self):
        """
        Initialize TanH activation layer.
        """
        super().__init__(trainable=False)
        self.input_tensor = None

    def forward(self, input_tensor):
        """
        Forward pass of TanH activation.

        Parameters:
        input_tensor (numpy.ndarray): Input tensor.

        Returns:
        numpy.ndarray: Output tensor after TanH activation.
        """
        self.input_tensor = input_tensor
        return np.tanh(input_tensor)

    def backward(self, error_tensor):
        """
        Backward pass of TanH activation.

        Parameters:
        error_tensor (numpy.ndarray): Error tensor from next layer.

        Returns:
        numpy.ndarray: Error tensor for previous layer.
        """
        # Derivative of tanh(x) is 1 - tanh²(x)
        tanh_output = np.tanh(self.input_tensor)
        derivative = 1 - np.square(tanh_output)
        return error_tensor * derivative