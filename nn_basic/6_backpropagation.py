import numpy as np
np.random.seed(1)
# input data 
inputs = np.array([[1,0,1], 
					[0,1,0], 
					[1,1,1],
					[1,0,0]])

# output data
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
		self.dinputs = np.dot(dvalues, self.weights.T)



# activation function
class Activation_Sigmoid:
	def forward(self, inputs):
		self.inputs = inputs
		self.output = 1/(1+np.exp(-1*inputs))
	def backward(self, dvalues):
		sigmoid = self.output
		d_sigmoid = (sigmoid)*(1-sigmoid)
		self.dinputs = np.multiply(dvalues, d_sigmoid)

# loss
class Loss:
	def forward(self, y_pred, y_true):
		self.output = np.subtract(y_true , y_pred)
		self.cost = 1/2*(self.output)*(self.output)
	def backward(self):
		self.dinputs = -1*self.output # (y_pred - y_true)

layer1 = Dense_Layer(3, 1)
activation1 = Activation_Sigmoid()
loss = Loss()

# update and optimizer
for i in range(1000):
    layer1.forward(inputs)
    activation1.forward(layer1.output)
    loss.forward(activation1.output, y_true)
   
    if(i%100==0):
        print(f'[{i}] cost\n{loss.cost}\n')

    loss.backward()
    activation1.backward(loss.dinputs)
    layer1.backward(activation1.dinputs)
    layer1.weights -= layer1.dweights

print(f'New synaptic weights after training:\n{layer1.weights}\n')


print('Considering new situation:\n{[0,1,1]}\n')
newZ = np.dot(np.array([0,1,1]), layer1.weights)
activationOutput = 1/(1+np.exp(-newZ))
print(activationOutput)


