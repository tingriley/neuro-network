import numpy as np
 
epochs = 1           # Number of iterations
inputLayerSize, hiddenLayerSize, outputLayerSize = 2, 3, 1
 
X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([ [0],   [1],   [1],   [0]])
 
def sigmoid (x): return 1/(1 + np.exp(-x))      # activation function
def sigmoid_(x): return x * (1 - x)             # derivative of sigmoid
                                                # weights on layer inputs
Wh = np.random.uniform(size=(inputLayerSize, hiddenLayerSize))
Wz = np.random.uniform(size=(hiddenLayerSize,outputLayerSize))
 
for i in range(epochs):
 
    H = sigmoid(np.dot(X, Wh))                  # hidden layer results
    print("Hidden Layer")
    print(H)

    Z = sigmoid(np.dot(H, Wz))                  # output layer results
    print("Output Layer")
    print(Z)

    E = Y - Z                                   # how much we missed (error)
    dZ = E * sigmoid_(Z)                        # delta Z
    dH = dZ.dot(Wz.T) * sigmoid_(H)             # delta H
    Wz +=  H.T.dot(dZ)                          # update output layer weights
    Wh +=  X.T.dot(dH)                          # update hidden layer weights

print("final result")
print(Z)  
