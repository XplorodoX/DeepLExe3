import numpy as np
import copy


class Optimizer:
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

    def copy(self):
        return copy.deepcopy(self)


class Sgd(Optimizer):
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weights = weight_tensor - self.learning_rate * gradient_tensor
        if self.regularizer is not None:
            updated_weights -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)
        return updated_weights


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(weight_tensor)

        self.velocity = self.momentum_rate * self.velocity - self.learning_rate * gradient_tensor
        updated_weights = weight_tensor + self.velocity

        if self.regularizer is not None:
            updated_weights -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        return updated_weights


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.velocity = None
        self.r = None
        self.t = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.velocity is None:
            self.velocity = np.zeros_like(gradient_tensor)
        if self.r is None:
            self.r = np.zeros_like(gradient_tensor)

        self.t += 1

        self.velocity = self.mu * self.velocity + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.square(gradient_tensor)

        v_hat = self.velocity / (1 - self.mu ** self.t)
        r_hat = self.r / (1 - self.rho ** self.t)

        updated_weights = weight_tensor - self.learning_rate * (v_hat / (np.sqrt(r_hat) + 1e-8))

        if self.regularizer is not None:
            updated_weights -= self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)

        return updated_weights