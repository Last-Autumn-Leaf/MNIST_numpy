from dnn_framework.optimizer import Optimizer
class SgdOptimizer(Optimizer):
    def __init__(self,parameters, learning_rate):
        super(SgdOptimizer, self).__init__(parameters)

        self.learning_rate=learning_rate

    def _step_parameter(self, parameter, parameter_grad, parameter_name):

        return parameter-self.learning_rate*parameter_grad