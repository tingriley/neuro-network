import torch

x = torch.randn(3, requires_grad = True)

x.requires_grad_(False)
print(x)
'''
y = x + 2
z = y*y*2
#z.mean()

v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)

print(z)
z.backward(v)

print(x.grad)'''


