import numpy as np
np.random.seed(1)
 

inputs = np.array([[0,0], [0,1], [1,0], [1,1]])
y_true = np.array([[0,1,1,0]]).T

class Convolution_Layer:
	def __init__(self, n_filters, filter_size): #stride: sliding window step, padding layers to be added
		# result size = ((nh-f)+1)((nw-f)+1)
		# result size = (n-f+2p+1), p = (f-1)/2, [(n-f+2p)/s+1], floor 
		# valid padding, same 
		self.n_filters = n_filters
		self.filter_size = filter_size
		self.conv_filter = np.random.randn(n_filters, filter_size, filter_size)/(filter_size*filter_size)
		print(self.conv_filter)
	def forward(self, inputs):
		pass
	def backward(self, dvalues):
		pass
		#self.dweights = np.dot(self.inputs.T, dvalues)
		#self.dinputs = np.dot(dvalues, self.weights.T)

c = Convolution_Layer(3,2)
