import numpy as np
np.random.seed(1)
 
inputs = np.array([[1,2,3,4], 
					[5,6,7,8], 
					[9,10,11,12]])


class Dense_Layer:
	def __init__(self, n_inputs, n_neurons):
		self.biases = np.zeros((1, n_neurons))
		self.weights = np.random.randn(n_inputs, n_neurons)
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
	def forward(self, inputs):
		self.inputs = inputs
		self.output = np.maximum(0, self.inputs)



layer1 = Dense_Layer(4, 3)
layer1.forward(inputs)

activation1 = Activation_ReLU()
activation1.forward(layer1.output)
print("Layer1: ")
print(layer1.output)
print("")

print("Activation_ReLU: ")
print(activation1.output)
print("")
