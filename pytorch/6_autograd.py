import torch
import torch.nn as nn

import numpy as np 

# 1 ) Design model (input, output, size, forward pass)
# 2 ) Construct loss and optimizer
# 3 ) Trainign loop
# - forward pass: compute prediction
# - backward pass: gradients
# - update weights

X = torch.tensor([1,2,3,4], dtype=torch.float32)
Y = torch.tensor([2,4,6,8], dtype=torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
	return w * x

print(f'prediction before training: f(5) = {forward(5):.3f}')

# Training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)


for epoch in range(n_iters):
	# prediction = forward pass
	y_pred = forward(X)

	#loss
	l = loss(Y, y_pred)

	#gradients
	l.backward()

	#update weights
	optimizer.step()

	#zero gradient
	optimizer.zero_grad()

	if epoch % 10 == 0:
		print(f'epoch {epoch + 1}: w = {w: .3f}, loss = {l:.8f}')



