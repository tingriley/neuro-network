import numpy as np
import torch
import torch.nn as nn

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
	def backward(self, dvalues):
		self.dweights = np.dot(self.inputs.T, dvalues)

layer1 = Dense_Layer(3, 1)
layer1.forward(inputs)
print(f'layer1.output\n {layer1.output}\n')

z = torch.from_numpy(layer1.output)
z.requires_grad_(True)
activation1 = torch.sigmoid(z)
print(f'activation1\n {activation1}\n')

ds = activation1.backward(z)
print(ds)


'''
for iteration in range(1000):
    layer1.forward(inputs)
    activation1.forward(layer1.output)
    loss.forward(activation1.output, y_true)
    print("cost")
    print(loss.cost)
    loss.backward(loss.output)
    activation1.backward(loss.dinputs)
    layer1.backward(activation1.dinputs)
    layer1.weights += layer1.dweights

print("New synaptic weights after training: ")
print(layer1.weights)

print("Considering new situation: [0,1,1]")
newZ = np.dot(np.array([0,1,1]), layer1.weights)
activationOutput = 1/(1+np.exp(-newZ))
print(activationOutput)
'''