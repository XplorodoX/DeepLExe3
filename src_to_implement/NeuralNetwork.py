import copy
import pickle


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self._phase = None

    @property
    def phase(self):
        """Get the current phase (training or testing)."""
        return self._phase

    @phase.setter
    def phase(self, value):
        """Set the phase for all layers."""
        self._phase = value
        for layer in self.layers:
            layer.testing_phase = value

    def forward(self):
        input_tensor, self.label_tensor = self.data_layer.next()
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        loss = self.loss_layer.forward(input_tensor, self.label_tensor)

        # Add regularization loss
        regularization_loss = 0.0
        for layer in self.layers:
            if hasattr(layer, 'optimizer') and layer.optimizer and hasattr(layer.optimizer,
                                                                           'regularizer') and layer.optimizer.regularizer:
                if hasattr(layer, 'weights'):
                    regularization_loss += layer.optimizer.regularizer.norm(layer.weights)
                # For RNN layer, also check for fc_hidden and fc_output weights
                if hasattr(layer, 'fc_hidden') and hasattr(layer.fc_hidden, 'weights'):
                    regularization_loss += layer.optimizer.regularizer.norm(layer.fc_hidden.weights)
                if hasattr(layer, 'fc_output') and hasattr(layer.fc_output, 'weights'):
                    regularization_loss += layer.optimizer.regularizer.norm(layer.fc_output.weights)

        return loss + regularization_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            # Erstellt eine tiefe Kopie, um sicherzustellen, dass jeder Layer
            # seinen eigenen Optimierer-Zustand hat (z.B. für Momentum oder Adam)
            layer.optimizer = self.optimizer.copy()
            layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            layer.testing_phase = True
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    def __getstate__(self):
        """
        Exclude data layer from pickling.
        """
        state = self.__dict__.copy()
        # Remove data layer as it contains generator objects
        state['data_layer'] = None
        return state

    def __setstate__(self, state):
        """
        Restore state and initialize dropped members.
        """
        self.__dict__.update(state)
        # Initialize dropped members with None
        if 'data_layer' not in state or state['data_layer'] is None:
            self.data_layer = None

    def save(filename, net):
        """
        Save neural network to file using pickle.

        Parameters:
        filename (str): Path to save file.
        net: Neural network to save.
        """
        with open(filename, 'wb') as f:
            pickle.dump(net, f)

    def load(filename, data_layer):
        """
        Load neural network from file and set data layer.

        Parameters:
        filename (str): Path to load file.
        data_layer: Data layer to set after loading.

        Returns:
        Neural network with data layer set.
        """
        with open(filename, 'rb') as f:
            net = pickle.load(f)
        net.data_layer = data_layer
        return net