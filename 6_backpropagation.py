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
	def backward(self, dvalues):
		self.dweights = np.dot(self.inputs.T, dvalues)

class Activation_ReLU:
	def forward(self, inputs):
		self.inputs = inputs
		self.output = np.maximum(0, self.inputs)
	def forward(self, dvalues):
		self.dinputs = dvalues.copy()
		self.dinputs[self.inputs <= 0] = 0


class Activation_Sigmoid:
	def forward(self, inputs):
		self.inputs = inputs
		self.output = 1/(1+np.exp(-1*inputs))
	def backward(self, dvalues):
		sigmoid = self.output
		d_sigmoid = (sigmoid)*(1-sigmoid)
		self.dinputs = np.multiply(dvalues, d_sigmoid)

class Loss:
	def forward(self, y_pred, y_true):
		self.output = np.subtract(y_true , y_pred)
		self.cost = 1/2*(self.output)*(self.output)
	def backward(self, dinputs):
		self.dinputs = dinputs

layer1 = Dense_Layer(3, 1)
activation1 = Activation_Sigmoid()
loss = Loss()


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
