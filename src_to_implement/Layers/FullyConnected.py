from .Base import BaseLayer
import numpy as np
class FullyConnected(BaseLayer):
    def __init__(self , input_size , output_size , trainable=True) -> None: #constructor for initialisation
        self._optimizer = None

        self.input_size = input_size
        self.output_size = output_size
        super().__init__(trainable) # inherit
        self.weights = np.random.rand(self.input_size+1 , self.output_size)  
        #uniform distribution -> range(0,1)
        # self.input_size+1 we are adding bias to the rows 

    
    def initialize(self, weights_initializer, bias_initializer):

        weights_shape = (self.input_size, self.output_size)
        bias_shape = (1,self.output_size)
    
        W = weights_initializer.initialize(weights_shape , self.input_size , self.output_size)
        b = bias_initializer.initialize(bias_shape , 1, self.output_size)
        self.weights = np.concatenate((W, b), axis=0)





    def forward(self , input_tensor):

        self.input_tensor = input_tensor

        bias_col = np.ones(  (self.input_tensor.shape[0] , 1) ) #tuple passed rows = batch size , no of col = 1
        

        self.new_input = np.concatenate((bias_col , self.input_tensor) , axis = 1) # adding 1 as a column 

        self.batch_size = self.input_tensor.shape[0] # rows of input tensor
        self.features_per_sample = self.input_tensor.shape[1] # columns of input tensor
        
        Z = np.dot( self.new_input , self.weights   ) # shape = no of samples (images) , no of neurons in current layer
        return Z
    
    
    @property # getter
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self,value):
        self._optimizer = value

    @property #getter
    def gradient_weights(self):
        return self._gradient_weights
    
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value


    def backward(self , error_tensor): # this error_tensor is dZ that came from reLU

        self.error_tensor_dZ = error_tensor

        self.gradient_weights = np.dot(self.new_input.T , error_tensor) #differentiate Z with respect to W (weights)

        if (self.optimizer is not None):
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)

        return np.dot(self.error_tensor_dZ , self.weights[1: , :].T) # gradient wrt input is passed back (    dl/dx = dl/dz(error tensor) * weights.T    )

        