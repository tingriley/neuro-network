import torch
import torch.nn as nn
import torch.nn.functional as F

weights = torch.ones(4, requires_grad = True) # required

optimizer = torch.optim.SGD(weights, lr = 0.01)
optimizer.step()
optimizer.zero_grad() # required


