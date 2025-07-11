import numpy as np
from .Base import BaseLayer
class Pooling(BaseLayer):
    def __init__(self , stride_shape , pooling_shape , trainable = False) -> None:
        super().__init__(trainable)
        self.stride_y = stride_shape[0]
        self.stride_x = stride_shape[1]
        self.pool_height = pooling_shape[0]
        self.pool_width = pooling_shape[1]



    def forward(self,input_tensor):
        if len(input_tensor.shape) == 4:
            self.input_tensor = input_tensor

            self.batch_size = input_tensor.shape[0]
            self.channels_input = input_tensor.shape[1]
            self.input_height = input_tensor.shape[2]
            self.input_width = input_tensor.shape[3]


            self.output_height = int (np.floor ((((self.input_height - self.pool_height))/self.stride_y) + 1 ))
            self.output_width = int (np.floor ((((self.input_width - self.pool_width))/self.stride_x) + 1 ))

            output_tensor = np.zeros((self.batch_size , self.channels_input , self.output_height , self.output_width))
            self.mask = np.zeros_like(input_tensor)

            for images in range(self.batch_size):
                for filter in range(self.channels_input):
                    for height in range(self.output_height):
                        for width in range(self.output_width):

                            vertical_start = height*self.stride_y
                            vertical_end = vertical_start + self.pool_height
                            horizontal_start = width*self.stride_x
                            horizontal_end = horizontal_start + self.pool_width
                            
                            window = input_tensor[images , filter , vertical_start: vertical_end , horizontal_start:horizontal_end]
                            output_tensor[images , filter , height , width] = np.max (window)


            return output_tensor
        

    

    def backward(self,error_tensor): # this is dL/doutput (of max pool forward pass)
        error_tensor_input = np.zeros_like(self.input_tensor)

        for images in range(self.batch_size):
                for filter in range(self.channels_input):
                    for height in range(self.output_height):
                        for width in range(self.output_width):

                            vertical_start = height*self.stride_y
                            vertical_end = vertical_start + self.pool_height
                            horizontal_start = width*self.stride_x
                            horizontal_end = horizontal_start + self.pool_width
                            
                           
                            window = self.input_tensor[images , filter , vertical_start: vertical_end , horizontal_start:horizontal_end]
                            max_val = np.max(window)
                            mask_local = (window == max_val)

                            error_tensor_input[images , filter , vertical_start: vertical_end , horizontal_start:horizontal_end] += mask_local * error_tensor[images , filter , height , width]

        return error_tensor_input