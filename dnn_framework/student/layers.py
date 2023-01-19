import numpy
import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    def __init__(self,I,J):
        self.I=I
        self.J=J
        self.parameters={}
        self.parameters['w']=np.random.normal(0, 2/(J+I), (J,I))
        self.parameters['b']=np.random.normal(0, 2/(J), J)
        super(FullyConnectedLayer, self).__init__()

    def get_parameters(self):
        return self.parameters
    def get_buffers(self):
        return {}

    def forward(self, x):
        y=np.matmul (x,self.get_parameters()['w'].transpose() ) +self.get_parameters()['b']
        cache=x.copy()
        return y,cache

    def backward(self, output_grad, cache):
        X=cache
        gradX = np.matmul(output_grad, self.parameters['w'])
        gradW = np.matmul(output_grad.transpose(), X)
        gradB = np.sum(output_grad, axis=0)

        dictionnary={
            "w":gradW,
            "b":gradB
        }
        return (gradX,dictionnary)
class BatchNormalization(Layer):
    def __init__(self,size):

        self.parameters={}
        self.buffers = {}
        self.parameters['gamma']=np.ones(size)
        self.parameters['beta']=np.zeros(size)
        self.buffers['global_mean']=np.zeros(size)
        self.buffers['global_variance']=np.zeros(size)

        self.batch_size = 0
        super().__init__()

    def forward(self, x):

        if self.is_training() :

            batch_mu=np.mean(x,0)
            batch_var=np.var(x,0)

            x_normalised=(x-batch_mu)/np.sqrt(batch_var+1e-11)
            y=self.parameters['gamma'] *x_normalised +self.parameters['beta']

            self.buffers['global_mean'] =batch_mu
            self.buffers['global_variance'] = batch_var
        else :

            e_mu=self.buffers['global_mean']
            e_var=self.buffers['global_variance']
            x_normalised = (x - e_mu) / np.sqrt(e_var + 1e-11)
            y = self.parameters['gamma'] * x_normalised + self.parameters['beta']

        return y,x_normalised.copy()

    def backward(self, output_grad, cache):
        grad=np.zeros_like(output_grad)
        dB=np.sum(output_grad,0)
        dG=np.sum(np.multiply(output_grad,cache),0)
        dictionnary = {
            "gamma": dG,
            "beta": dB
        }
        return (grad,dictionnary)

    def get_buffers(self):
        return self.buffers
    def get_parameters(self):
        return self.parameters

class Sigmoid(Layer):
    
    def __init__(self):
        super(Sigmoid, self).__init__()
    
    def forward(self, x):
        return 1 / (1 + np.exp(-x)),x.copy()

    def backward(self, output_grad, cache):
        grad = output_grad.copy()
        grad *= self.forward(cache)[0] * (1 - self.forward(cache)[0])
        return (grad,None)

    def get_parameters(self):
        return {}
    def get_buffers(self):
        return {}
class ReLU(Layer):
    
    def __init__(self):
        super(ReLU, self).__init__()
    
    def forward(self, x):
        Y = x.copy()
        Y[Y < 0] = 0
        return Y,x.copy()

    def backward(self, output_grad, cache):
        grad = output_grad.copy()
        grad[cache < 0] = 0
        return (grad, {})

    def get_parameters(self):
        return {}
    def get_buffers(self):
        return {}