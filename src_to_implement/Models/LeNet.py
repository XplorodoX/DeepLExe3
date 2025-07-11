import numpy as np
from NeuralNetwork import NeuralNetwork
from Layers.FullyConnected import FullyConnected
from Layers.Conv import Conv
from Layers.Pooling import Pooling
from Layers.Flatten import Flatten
from Layers.ReLU import ReLU
from Layers.SoftMax import SoftMax
from Optimizers import Adam
from Regularization import L2

class LeNet:
    """
    LeNet architecture implementation.
    """

    def __init__(self):
        """
        Initialize LeNet network.
        """
        self.network = self.build()

    def build(self):
        """
        Build LeNet architecture.

        Returns:
        NeuralNetwork: Configured LeNet network.
        """
        # Create network
        net = NeuralNetwork()

        # L2 regularizer
        regularizer = L2(4e-4)

        # ADAM optimizer
        optimizer = Adam(5e-4, 0.9, 0.999)
        optimizer.add_regularizer(regularizer)

        # Layer 1: Convolution + ReLU
        conv1 = Conv(stride_shape=(1, 1), convolution_shape=(5, 5), num_kernels=6)
        conv1.optimizer = optimizer
        conv1.initialize()
        net.append_layer(conv1)
        net.append_layer(ReLU())

        # Layer 2: Average Pooling
        net.append_layer(Pooling(stride_shape=(2, 2), pooling_shape=(2, 2)))

        # Layer 3: Convolution + ReLU
        conv2 = Conv(stride_shape=(1, 1), convolution_shape=(5, 5), num_kernels=16)
        conv2.optimizer = optimizer
        conv2.initialize()
        net.append_layer(conv2)
        net.append_layer(ReLU())

        # Layer 4: Average Pooling
        net.append_layer(Pooling(stride_shape=(2, 2), pooling_shape=(2, 2)))

        # Flatten for fully connected layers
        net.append_layer(Flatten())

        # Layer 5: Fully Connected + ReLU
        fc1 = FullyConnected(input_size=400, output_size=120)
        fc1.optimizer = optimizer
        fc1.initialize()
        net.append_layer(fc1)
        net.append_layer(ReLU())

        # Layer 6: Fully Connected + ReLU
        fc2 = FullyConnected(input_size=120, output_size=84)
        fc2.optimizer = optimizer
        fc2.initialize()
        net.append_layer(fc2)
        net.append_layer(ReLU())

        # Layer 7: Fully Connected + SoftMax
        fc3 = FullyConnected(input_size=84, output_size=10)
        fc3.optimizer = optimizer
        fc3.initialize()
        net.append_layer(fc3)
        net.append_layer(SoftMax())

        return net

    def forward(self, input_tensor):
        """
        Forward pass through the network.

        Parameters:
        input_tensor (numpy.ndarray): Input tensor.

        Returns:
        numpy.ndarray: Output predictions.
        """
        return self.network.forward(input_tensor)

    def backward(self, error_tensor):
        """
        Backward pass through the network.

        Parameters:
        error_tensor (numpy.ndarray): Error tensor.

        Returns:
        numpy.ndarray: Gradient tensor.
        """
        return self.network.backward(error_tensor)