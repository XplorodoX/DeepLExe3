import numpy as np
from .Base import BaseLayer


class Dropout(BaseLayer):
    """
    Dropout layer for regularization.
    """

    def __init__(self, probability):
        """
        Initialize Dropout layer.

        Parameters:
        probability (float): Fraction of units to keep (not drop).
        """
        super().__init__(trainable=False)
        self.probability = probability
        self.mask = None
        self.testing_phase = False  # Initialize testing_phase

    def forward(self, input_tensor):
        """
        Forward pass of Dropout layer.

        Parameters:
        input_tensor (numpy.ndarray): Input tensor.

        Returns:
        numpy.ndarray: Output tensor after dropout.
        """
        if self.testing_phase:
            # During testing, use all units but scale by probability
            return input_tensor
        else:
            # During training, randomly drop units
            self.mask = np.random.binomial(1, self.probability, input_tensor.shape) / self.probability
            return input_tensor * self.mask

    def backward(self, error_tensor):
        """
        Backward pass of Dropout layer.

        Parameters:
        error_tensor (numpy.ndarray): Error tensor from next layer.

        Returns:
        numpy.ndarray: Error tensor for previous layer.
        """
        # Apply the same mask used in forward pass
        return error_tensor * self.mask