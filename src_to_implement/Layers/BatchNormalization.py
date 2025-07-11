import numpy as np
from .Base import BaseLayer
from .Helpers import compute_bn_gradients

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__(trainable=True)
        self.channels = channels
        self.initialize()
        self.optimizer_weights = None
        self.optimizer_bias = None
        self.moving_mean = None
        self.moving_var = None
        self.alpha = 0.8

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)


    def reformat(self, tensor):
        if tensor.ndim == 4:
            self._stored_shape = tensor.shape
            batch_size, channels, height, width = tensor.shape
            reshaped = tensor.permute(0, 2, 3, 1) if hasattr(tensor, 'permute') else np.transpose(tensor, (0, 2, 3, 1))
            return reshaped.reshape(-1, channels)
        else:
            batch_size, channels, height, width = self._stored_shape
            reshaped = tensor.reshape(batch_size, height, width, channels)
            return np.transpose(reshaped, (0, 3, 1, 2))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        # Handle 4D tensors (convolutional case)
        if input_tensor.ndim == 4:
            self._original_shape = input_tensor.shape
            input_tensor = self.reformat(input_tensor)

        if self.testing_phase:
            # Use moving averages for testing
            if self.moving_mean is None or self.moving_var is None:
                self.mean = np.mean(input_tensor, axis=0)
                self.var = np.var(input_tensor, axis=0)
            else:
                self.mean = self.moving_mean
                self.var = self.moving_var
        else:
            # Training phase: compute batch statistics
            self.mean = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)

            # Update moving averages
            if self.moving_mean is None:
                self.moving_mean = self.mean.copy()
                self.moving_var = self.var.copy()
            else:
                self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * self.mean
                self.moving_var = self.alpha * self.moving_var + (1 - self.alpha) * self.var

        self.normalized = (input_tensor - self.mean) / np.sqrt(self.var + np.finfo(float).eps)
        output = self.weights * self.normalized + self.bias

        # Reformat back to original shape if needed
        if hasattr(self, '_original_shape'):
            output = self.reformat(output)

        return output

    def backward(self, error_tensor):
        if error_tensor.ndim == 4:
            error_tensor = self.reformat(error_tensor)

        input_tensor = self.input_tensor
        if input_tensor.ndim == 4:
            input_tensor = self.reformat(input_tensor)

        # Compute gradients w.r.t. weights and bias
        self.gradient_weights = np.sum(error_tensor * self.normalized, axis=0)
        self.gradient_bias = np.sum(error_tensor, axis=0)

        # Update weights and bias if optimizers are available
        if self.optimizer_weights is not None:
            self.weights = self.optimizer_weights.calculate_update(self.weights, self.gradient_weights)

        if self.optimizer_bias is not None:
            self.bias = self.optimizer_bias.calculate_update(self.bias, self.gradient_bias)

        # Compute gradient w.r.t. input using helper function
        gradient_input = compute_bn_gradients(error_tensor, input_tensor, self.weights, self.mean, self.var)

        # Reformat back to original shape if needed
        if hasattr(self, '_original_shape'):
            gradient_input = self.reformat(gradient_input)

        return gradient_input


    @property
    def optimizer(self):
        return self.optimizer_weights

    @optimizer.setter
    def optimizer(self, optimizer):
        self.optimizer_weights = optimizer
        self.optimizer_bias = optimizer