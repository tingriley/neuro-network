import numpy as np
import matplotlib.pyplot as plt

def Sigmoid(x):
    ''' It returns 1/(1+exp(-x)). where the values lies between zero and one '''
    return 1/(1+np.exp(-x))


def ReLU(x):
    ''' It returns zero if the input is less than zero otherwise it returns the given input. '''
    output = []
    for i in x:
        output.append(max(i,0))
    return output

def Softmax(x):
    ''' Compute softmax values for each sets of scores in x. '''
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def Plot(title, name):
	x = np.linspace(-10, 10)
	if name == 'ReLU':
		plt.plot(x, ReLU(x))
	elif name == 'Sigmoid':
		plt.plot(x, Sigmoid(x))
	elif name == 'Softmax':
		plt.plot(x, Softmax(x))
	plt.axis('tight')
	plt.title(title)
	plt.savefig(name + '.png')
	plt.close()

Plot('Activation Function: ReLU', 'ReLU')
Plot('Activation Function: Sigmoid', 'Sigmoid')
Plot('Activation Function: Softmax', 'Softmax')

