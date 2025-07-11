from .Base import BaseLayer
import numpy as np

class ReLU(BaseLayer):

    def __init__(self, trainable = False):
        super().__init__(trainable)



    def forward(self,input_tensor): #Z from FC layer
        self.input_tensor = input_tensor
        batch_size = self.input_tensor.shape[0]#rows
        features_per_sample = self.input_tensor.shape[1]#columns

        relu_output = np.maximum(0,input_tensor)
        return relu_output
    
    def backward(self,error_tensor): # this error tensor we got from FC layer
        return error_tensor * (self.input_tensor > 0) # this is the error tensor wrt the input to relu layer (dl/dz)
