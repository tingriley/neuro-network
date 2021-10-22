import numpy as np
np.random.seed(1)

''' single neuron '''
inputs = np.array([1,2,3])
weights = np.array([0.2, 0.8, -0.5])
bias = 2

output = np.dot(inputs, weights.T) + bias
print("Example of single neuron")
print(output)
print("")


''' a layer of neurons '''
inputs = np.array([1,2,3,4])
weights = np.array([[0.2, 0.8, -0.5, 1.0], 
					[0.5, -0.91, 0.26, 0.7], 
					[-0.26, -0.27, 0.17, 0.9]])
bias = np.array([2,3,4])

output = np.dot(inputs, weights.T) + bias
print("Example of a layer of neurons")
print(output)
print("")

import numpy as np
''' a batch of input '''
inputs = np.array([[1,2,3,4], 
					[5,6,7,8], 
					[9,10,11,12]])
weights = np.array([[0.2, 0.8, -0.5, 1.0], 
					[0.5, -0.91, 0.26, 0.7], 
					[-0.26, -0.27, 0.17, 0.9]])
bias = np.array([2,3,4])

output = np.dot(inputs, weights.T) + bias
print("Example of a batch of data")
print(output)
print("")

