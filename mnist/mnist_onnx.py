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

epochs = 1

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


''' create model '''
# Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
           nn.Conv2d(
               in_channels = 1,
               out_channels = 16,
               kernel_size = 5,
               stride = 1,
               padding = 2
           ),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # fully connected layer, output 10 classes
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # flatten
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output, x


cnn = CNN().to(device)
print(cnn)

# define Loss function
criterion = nn.CrossEntropyLoss()

# define optimizer function
optimizer = optim.Adam(cnn.parameters(), lr = 0.01)
print(optimizer)

# Training

# Training
for epoch in range(epochs):
    running_loss = 0.0

    for times, data in enumerate(trainLoader):
       
        inputs, labels = data[0].to(device), data[1].to(device)
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Foward + backward + optimize
        outputs = cnn(inputs)

        loss = criterion(outputs[0], labels) # outputs[100, 10]
        loss.backward()
        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        if times % 100 == 99 or times+1 == len(trainLoader):
            print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, times+1, len(trainLoader), running_loss/2000))

print('Training Finished.')

# Testing
correct = 0
total = 0

with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
 

        outputs = cnn(inputs)
        _, predicted = torch.max(outputs[0].data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct / total))

class_correct = [0 for i in range(10)]
class_total = [0 for i in range(10)]

with torch.no_grad():
    for data in testLoader:
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = cnn(inputs)
        _, predicted = torch.max(outputs[0], 1)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
            #print(class_correct)
            #print(class_total)

for i in range(10):
    print('Accuracy of %d: %3f' % (i, (class_correct[i]/class_total[i])))




device = T.device("cpu")

torch.save(cnn.state_dict(),"./mnist.pt")

trained_model = CNN()
trained_model.load_state_dict(torch.load('./mnist.pt'))
dummy_input = Variable(torch.randn(1, 1, 28, 28)) 
torch.onnx.export(trained_model, dummy_input, "./mnist.onnx", input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})

