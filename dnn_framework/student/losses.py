from dnn_framework.loss import Loss
import numpy as np
class CrossEntropyLoss(Loss):
    def __init__(self):
        super().__init__()

    def calculate(self, x, target):
        m,n = x.shape[0],x.shape[1]
        p = softmax(x)
        log_likelihood = -np.log(p[range(m), target])
        loss = np.sum(log_likelihood) / (m)

        grad = softmax(x)
        grad[range(m), target] -= 1
        grad = grad / (m)

        return (loss,grad)


def softmax(X):
    exps = np.exp(X - X.max())
    return exps / np.sum(exps,1,keepdims=True)


class MeanSquaredErrorLoss:
    def __init__(self):
        super().__init__()

    def calculate(self, x, target):
        return (np.square(x - target)).mean(),2*(x-target)/(x.shape[0]*x.shape[1])