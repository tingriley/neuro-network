import numpy as np

inputs = np.array([[1,2,3,4], 
					[5,6,7,8], 
					[9,10,11,12]])


class Dense_Layer:
	def __init__(self, n_inputs, n_neurons):
		self.biases = np.zeros((1, n_neurons))
		print(self.biases)
		self.weights = np.random.randn(n_inputs, n_neurons)
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weights) + self.biases


layer1 = Dense_Layer(4, 3)
layer1.forward(inputs)
print(layer1.output)

