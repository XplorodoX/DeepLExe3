import numpy as np
from .Base import BaseLayer
class Flatten(BaseLayer):
    def __init__(self , trainable = False):
        super().__init__(trainable)

    def forward(self , input_tensor):
        self.input_tensor = input_tensor
        input_tensor_transform = np.array(input_tensor)
        return input_tensor_transform.reshape(input_tensor_transform.shape[0] , -1)
    
    def backward(self , error_tensor):
        return error_tensor.reshape(self.input_tensor.shape)
        

    