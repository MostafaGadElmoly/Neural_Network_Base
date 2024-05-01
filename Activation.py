import numpy as np
import nnfs

# Importing dataset from nnfs library
from nnfs.datasets import spiral_data

# Initializing nnfs library
nnfs.init()

# Sample input data
X = [[1, 2, 3, 2.5],
     [2.0, 5.5, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

# Generating spiral dataset with 100 samples and 3 classes
X, Y = spiral_data(100, 3)

# Class for dense layer
class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        # Initializing weights randomly and biases to zero
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Forward pass through the layer
        self.output = np.dot(inputs, self.weights) + self.biases

# Class for ReLU activation function
class ActivationReLU:
    def forward(self, inputs):
        # Forward pass through ReLU activation function
        self.output = np.maximum(0, inputs)

# Creating a dense layer with 2 input features and 5 neurons
layer1 = LayerDense(2, 5)

# Creating ReLU activation function
activation1 = ActivationReLU()

# Forward pass through the first layer
layer1.forward(X)

# Forward pass through the activation function
activation1.forward(layer1.output)

# Printing the output after activation
print(activation1.output)
