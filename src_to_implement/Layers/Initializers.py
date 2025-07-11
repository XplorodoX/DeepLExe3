import numpy as np
class Constant:
    def __init__(self , value = 0.1) -> None:
        self.value = value

    def initialize(self , weights_shape , fan_in , fan_out):
        return np.full(weights_shape , self.value)



class UniformRandom:
    def __init__(self) -> None:
        pass

    def initialize(self , weights_shape , fan_in , fan_out):

        return np.random.rand(*weights_shape)


class Xavier:
    def __init__(self) -> None:
        pass

    def initialize(self , weights_shape , fan_in , fan_out):

        sd = np.sqrt(2/ (fan_in + fan_out))
        return np.random.normal(loc=0.0 , scale=sd , size=weights_shape)



class He:
    def __init__(self) -> None:
        pass


    def initialize(self , weights_shape , fan_in , fan_out):
        sd = np.sqrt(2/ (fan_in ))
        return np.random.normal(loc=0.0 , scale=sd , size=weights_shape)

