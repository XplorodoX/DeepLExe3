import numpy as np
from .Base import BaseLayer
from .FullyConnected import FullyConnected


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self._memorize = False
        self.fc_hidden = FullyConnected(hidden_size + input_size, hidden_size)
        self.fc_output = FullyConnected(hidden_size, output_size)
        self.input_cache = []
        self.hidden_cache = []
        self.concatenated_cache = []
        self._optimizer = None
        self.batch_size = None

    @property
    def memorize(self):
        """Get memorize state."""
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        """Set memorize state."""
        self._memorize = value
        if not value:
            # Reset hidden state when memorize is disabled
            self.hidden_state = np.zeros(self.hidden_size)

    @property
    def weights(self):
        """Get weights as stacked tensor."""
        return self.fc_hidden.weights

    @weights.setter
    def weights(self, weights):
        """Set weights from stacked tensor."""
        self.fc_hidden.weights = weights

    @property
    def gradient_weights(self):
        """Get gradient weights."""
        return self.fc_hidden.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        """Set gradient weights."""
        self.fc_hidden.gradient_weights = value

    @property
    def optimizer(self):
        """Get optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Set optimizer for both FC layers."""
        self._optimizer = optimizer
        self.fc_hidden.optimizer = optimizer
        self.fc_output.optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):
        self.fc_hidden.initialize(weights_initializer, bias_initializer)
        self.fc_output.initialize(weights_initializer, bias_initializer)

    def calculate_regularization_loss(self):
        loss = 0.0
        if self.optimizer and self.optimizer.regularizer:
            loss += self.optimizer.regularizer.norm(self.fc_hidden.weights)
            loss += self.optimizer.regularizer.norm(self.fc_output.weights)
        return loss

    def forward(self, input_tensor):
        batch_size, _ = input_tensor.shape
        self.batch_size = batch_size  # Store batch_size for backward pass
        self.input_tensor = input_tensor  # Store input for backward pass
        self.concatenated_cache = []
        self.hidden_cache = []
        outputs = []

        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)

        # Cache h_{-1}
        self.hidden_cache.append(self.hidden_state.copy())

        for t in range(batch_size):
            x_t = input_tensor[t:t + 1]
            h_prev = self.hidden_state.reshape(1, -1)

            concatenated = np.concatenate([h_prev, x_t], axis=1)
            self.concatenated_cache.append(concatenated)

            # Compute new hidden state h_t
            hidden_output = self.fc_hidden.forward(concatenated)
            self.hidden_state = np.tanh(hidden_output.flatten())
            self.hidden_cache.append(self.hidden_state.copy())  # Cache h_t

            # Compute output y_t
            output = self.fc_output.forward(self.hidden_state.reshape(1, -1))
            outputs.append(output)

        return np.vstack(outputs)

    def backward(self, error_tensor):
        batch_size = error_tensor.shape[0]
        input_error = np.zeros((batch_size, self.input_size))

        # Initialize accumulated gradients to zero
        accumulated_hidden_gradient_weights = np.zeros_like(self.fc_hidden.weights)
        accumulated_output_gradient_weights = np.zeros_like(self.fc_output.weights)

        hidden_error = np.zeros(self.hidden_size)

        # Backward pass through time
        for t in reversed(range(batch_size)):
            # Backward through output layer
            # Set the input that was used during forward pass at time t
            self.fc_output.input_tensor = self.hidden_cache[t + 1].reshape(1, -1)

            # Create bias column for FC output layer (since it expects it)
            bias_col = np.ones((1, 1))
            self.fc_output.new_input = np.concatenate((bias_col, self.fc_output.input_tensor), axis=1)

            # Compute output layer gradients
            output_gradient_weights = np.dot(self.fc_output.new_input.T, error_tensor[t:t + 1])
            accumulated_output_gradient_weights += output_gradient_weights

            # Compute error to propagate back from output layer
            output_error = np.dot(error_tensor[t:t + 1], self.fc_output.weights[1:, :].T)

            # Add hidden error from next timestep
            total_hidden_error = output_error.flatten() + hidden_error

            # Backward through tanh activation
            tanh_derivative = 1 - self.hidden_cache[t + 1] ** 2
            hidden_grad = total_hidden_error * tanh_derivative

            # Backward through hidden layer
            # Set the input that was used during forward pass at time t
            self.fc_hidden.input_tensor = self.concatenated_cache[t]

            # Create bias column for FC hidden layer
            bias_col = np.ones((1, 1))
            self.fc_hidden.new_input = np.concatenate((bias_col, self.fc_hidden.input_tensor), axis=1)

            # Compute hidden layer gradients
            hidden_gradient_weights = np.dot(self.fc_hidden.new_input.T, hidden_grad.reshape(1, -1))
            accumulated_hidden_gradient_weights += hidden_gradient_weights

            # Compute error to propagate back from hidden layer
            concatenated_error = np.dot(hidden_grad.reshape(1, -1), self.fc_hidden.weights[1:, :].T)

            # Split error between hidden state and input
            hidden_error = concatenated_error[0, :self.hidden_size]
            input_error[t] = concatenated_error[0, self.hidden_size:]

        # Set the accumulated gradients
        self.fc_hidden.gradient_weights = accumulated_hidden_gradient_weights
        self.fc_output.gradient_weights = accumulated_output_gradient_weights

        # Update weights if optimizers are available
        if self.fc_hidden.optimizer is not None:
            self.fc_hidden.weights = self.fc_hidden.optimizer.calculate_update(
                self.fc_hidden.weights, self.fc_hidden.gradient_weights)

        if self.fc_output.optimizer is not None:
            self.fc_output.weights = self.fc_output.optimizer.calculate_update(
                self.fc_output.weights, self.fc_output.gradient_weights)

        return input_error