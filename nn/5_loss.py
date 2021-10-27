import numpy as np
np.random.seed(1)
 
inputs = np.array([[1,0,1], 
					[0,1,0], 
					[1,1,1],
					[1,0,0]])

y_true = np.array([[1,0,1,0]]).T

class Dense_Layer:
	def __init__(self, n_inputs, n_neurons):
		self.biases = np.zeros((1, n_neurons))
		self.weights = np.random.random((n_inputs, n_neurons))
	def forward(self, inputs):
		self.inputs = inputs
		self.output = (np.dot(inputs, self.weights) + self.biases)


class Activation_Sigmoid:
	def forward(self, inputs):
		self.inputs = inputs
		self.output = 1/(1+np.exp(-1*inputs))

class Loss:
	def forward(self, y_pred, y_true):
		self.output = np.subtract(y_true , y_pred)
		self.cost = 1/2*(self.output)*(self.output)

layer1 = Dense_Layer(3, 1)
layer1.forward(inputs)
activation1 = Activation_Sigmoid()
activation1.forward(layer1.output)
loss = Loss()
loss.forward(activation1.output, y_true)

print(f'loss:\n{loss.cost}\n')
