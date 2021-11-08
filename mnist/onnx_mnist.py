import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dset
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx
import numpy as np

import onnx
import onnxruntime
import torch as T

class_correct = [0 for i in range(10)]
class_total = [0 for i in range(10)]

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Transform
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),]
)
# Data
trainSet = datasets.MNIST(root='MNIST', download=True, train=True, transform=transform)
testSet = datasets.MNIST(root='MNIST', download=True, train=False, transform=transform)
trainLoader = dset.DataLoader(trainSet, batch_size=100, shuffle=True)
testLoader = dset.DataLoader(testSet, batch_size=100, shuffle=False)


onnx_model = onnx.load("simple_cnn.onnx")
onnx.checker.check_model(onnx_model)

ort_session = onnxruntime.InferenceSession("simple_cnn.onnx")

print(ort_session.get_inputs()[0])

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

with torch.no_grad():
    for data in testLoader:
        inputs, labels = data[0].to(device), data[1].to(device)

       #ort_inputs = {ort_session.get_inputs()[0].name: np.array(inputs)}
        #outputs  = ort_session.run(None, ort_inputs)

        ort_inputs = {ort_session.get_inputs()[0].name: np.array(inputs)}
        detections  = ort_session.run(None, ort_inputs)


        _, predicted = torch.max(torch.tensor(detections[0]), 1)
      
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %d: %3f' % (i, (class_correct[i]/class_total[i])))
