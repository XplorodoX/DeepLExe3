import numpy as np
class CrossEntropyLoss:

    def __init__(self) -> None:
        pass

    def forward(self, prediction_tensor , label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor # one hot encoded 1 for the correct class and 0 everywhere else
        epsilon = np.finfo(float).eps


        self.batch_size = label_tensor.shape[0]

        correct_class_probs = np.sum(prediction_tensor * label_tensor, axis=1)

        loss_per_sample = -np.log(correct_class_probs + epsilon) # to avoid log(0)

        total_loss = np.sum(loss_per_sample) # loss for all the samples in my network

        return total_loss


    def backward(self, label_tensor):
        temp = -1/(self.prediction_tensor)
        return (temp*label_tensor)
