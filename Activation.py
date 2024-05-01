import numpy as np
import nnfs

from nnfs.datasets import spiral_data


nnfs.init()

X = [[1,2,3,2.5],
     [2.0,5.5,-1.0,2.0],
     [-1.5,2.7,3.3,-0.8]]

X,Y = spiral_data(100,3)

class layer_dense:
    def __init__(self,n_inputs, n_neurons):
        self.wegiht = 0.10*np.random.rand(n_inputs,n_neurons)
        self.bias = np.zeros((1, n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs, self.wegiht) + self.bias


class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)



layer1 = layer_dense(2,5)
Activation1 = Activation_ReLU()
layer1.forward(X)
#print(layer1.output)
Activation1.forward(layer1.output)
print(Activation1.output)


        
        
