import numpy as np
from .Base import BaseLayer
class SoftMax(BaseLayer):
    def __init__(self) -> None:

        self.trainable = False

    def forward(self , input_tensor):
        self.batch_size = input_tensor.shape[0] # rows represent the no of samples 
        self.features_per_sample = input_tensor.shape[1] # columns represent features in 1 sample in this case that is equal to number of classes
        self.input_tensor = input_tensor

        max_num_from_each_sample = np.max(self.input_tensor , axis = 1, keepdims = True) # rows stay the same
        scaled_input = self.input_tensor - max_num_from_each_sample


        exponential = np.exp(scaled_input)
        sum_per_row = np.sum(exponential , axis= 1 , keepdims= True)

        self.final_output = exponential / sum_per_row
        return self.final_output
    
    def backward(self , error_tensor):
        # i == j [   y_i*(1-y_i)   ]

        # i != j -y_i * y_j 
 
        final_grad = np.zeros_like(self.final_output)

        for individual_sample in range(self.batch_size):

            jacobian = np.diag(self.final_output[individual_sample , :]) - np.outer(self.final_output[individual_sample , :] ,  self.final_output[individual_sample , :])
            
            final_grad[individual_sample , :] = np.dot(jacobian , error_tensor[individual_sample , :])

        return final_grad
