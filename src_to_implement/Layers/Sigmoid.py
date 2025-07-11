import numpy as np
from .Base import BaseLayer


class Sigmoid(BaseLayer):
    """
    Sigmoid activation layer.
    """

    def __init__(self):
        """
        Initialize Sigmoid activation layer.
        """
        super().__init__(trainable=False)
        self.output_tensor = None

    def forward(self, input_tensor):
        """
        Forward pass of Sigmoid activation.

        Parameters:
        input_tensor (numpy.ndarray): Input tensor.

        Returns:
        numpy.ndarray: Output tensor after Sigmoid activation.
        """
        # Clip input to prevent overflow in exp
        clipped_input = np.clip(input_tensor, -500, 500)
        self.output_tensor = 1 / (1 + np.exp(-clipped_input))
        return self.output_tensor

    def backward(self, error_tensor):
        """
        Backward pass of Sigmoid activation.

        Parameters:
        error_tensor (numpy.ndarray): Error tensor from next layer.

        Returns:
        numpy.ndarray: Error tensor for previous layer.
        """
        # Derivative of sigmoid(x) is sigmoid(x) * (1 - sigmoid(x))
        derivative = self.output_tensor * (1 - self.output_tensor)
        return error_tensor * derivative