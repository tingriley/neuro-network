import torch
import torch.nn as nn
import torch.nn.functional as F

weights = torch.ones(4, requires_grad = True)

for epoch in range(3):
    model_output = (weights*3).sum()

    model_output.backward()

    print(weights.grad)

    weights.grad.zero_()
