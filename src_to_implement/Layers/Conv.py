import numpy as np
from .Base import BaseLayer
import copy

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels, trainable=True):
        super().__init__(trainable)
        self.stride = stride_shape
        self.num_kernels = num_kernels
        self._optimizer_weights = None
        self._optimizer_bias = None

        if len(convolution_shape) == 2:
            self.channels_filter = convolution_shape[0]
            self.filter_width = convolution_shape[1]
            self.filter_height = 1
        else:
            self.channels_filter = convolution_shape[0]
            self.filter_height = convolution_shape[1]
            self.filter_width = convolution_shape[2]

        self.weights = np.random.rand(num_kernels, self.channels_filter, self.filter_height, self.filter_width)
        self.bias = np.random.rand(num_kernels)

        if len(stride_shape) == 2:
            self.stride_y = stride_shape[0]
            self.stride_x = stride_shape[1]
        else:
            self.stride_y = self.stride_x = stride_shape[0]

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.channels_filter * self.filter_height * self.filter_width
        fan_out = self.num_kernels * self.filter_height * self.filter_width
        shape = (self.num_kernels, self.channels_filter, self.filter_height, self.filter_width)
        self.weights = weights_initializer.initialize(shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels,), fan_in, fan_out)

    def forward(self, input_tensor):
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, axis=2)
            self.was_1d = True
        else:
            self.was_1d = False

        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        self.channels_input = input_tensor.shape[1]
        self.input_height = input_tensor.shape[2]
        self.input_width = input_tensor.shape[3]

        self.pad_total_y = self.filter_height - 1
        self.pad_top = self.pad_total_y // 2
        self.pad_bottom = self.pad_total_y - self.pad_top

        self.pad_total_x = self.filter_width - 1
        self.pad_left = self.pad_total_x // 2
        self.pad_right = self.pad_total_x - self.pad_left

        self.input_pad = np.pad(input_tensor,
                                ((0, 0), (0, 0),
                                 (self.pad_top, self.pad_bottom),
                                 (self.pad_left, self.pad_right)),
                                mode='constant', constant_values=0)

        output_height = ((self.input_height + self.pad_total_y - self.filter_height) // self.stride_y) + 1
        output_width = ((self.input_width + self.pad_total_x - self.filter_width) // self.stride_x) + 1

        output_tensor = np.zeros((self.batch_size, self.num_kernels, output_height, output_width))

        for images in range(self.batch_size):
            for filter in range(self.num_kernels):
                for height in range(output_height):
                    for width in range(output_width):
                        v_start = height * self.stride_y
                        v_end = v_start + self.filter_height
                        h_start = width * self.stride_x
                        h_end = h_start + self.filter_width

                        output_tensor[images, filter, height, width] = np.sum(
                            self.input_pad[images, :, v_start:v_end, h_start:h_end] *
                            self.weights[filter]) + self.bias[filter]

        if self.was_1d:
            output_tensor = np.squeeze(output_tensor, axis=2)

        return output_tensor

    def backward(self, error_tensor):
        if error_tensor.ndim == 3:
            error_tensor = np.expand_dims(error_tensor, axis=2)

        bias_gradient = np.sum(error_tensor, axis=(0, 2, 3))
        weight_gradient = np.zeros_like(self.weights)
        input_gradient = np.zeros_like(self.input_pad)

        error_tensor_height = error_tensor.shape[2]
        error_tensor_width = error_tensor.shape[3]

        for images in range(self.batch_size):
            for filter in range(self.num_kernels):
                for height in range(error_tensor_height):
                    for width in range(error_tensor_width):
                        v_start = height * self.stride_y
                        v_end = v_start + self.filter_height
                        h_start = width * self.stride_x
                        h_end = h_start + self.filter_width

                        input_slice = self.input_pad[images, :, v_start:v_end, h_start:h_end]
                        error = error_tensor[images, filter, height, width]

                        input_gradient[images, :, v_start:v_end, h_start:h_end] += self.weights[filter] * error
                        weight_gradient[filter] += input_slice * error

        if self.pad_total_y > 0 or self.pad_total_x > 0:
            input_gradient = input_gradient[:, :,
                                            self.pad_top:self.pad_top + self.input_height,
                                            self.pad_left:self.pad_left + self.input_width]

        if self.was_1d:
            input_gradient = np.squeeze(input_gradient, axis=2)

        self.gradient_weights = weight_gradient
        self.gradient_bias = bias_gradient
        self._gradient_weights = weight_gradient
        self._gradient_bias = bias_gradient

        if self._optimizer_weights:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
        if self._optimizer_bias:
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        return input_gradient

    @property
    def optimizer(self):
        return self._optimizer_weights

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer_weights = copy.deepcopy(optimizer)
        self._optimizer_bias = copy.deepcopy(optimizer)