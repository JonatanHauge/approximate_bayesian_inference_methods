import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torch.optim as optim
from models.LeNet5 import LeNet
import numpy as np



# Load MNIST dataset and preprocess
mnist_train = MNIST('./datasets', train=True, download=False)
xtrain = mnist_train.data.float() / 255
xtrain = xtrain.unsqueeze(1)
ytrain = mnist_train.targets

'''
def extract_parameters(model):
    params = []
    for name, param in model.named_parameters():
        if param is None:
            continue
        params.append((model, name, param))
    return params
'''

def extract_parameters(model):
    params = []	
    for module in model.modules():	
        for name in list(module._parameters.keys()):	
            if module._parameters[name] is None:	
                continue	
            param = module._parameters[name]	
            params.append((module, name, param.size()))	
            module._parameters.pop(name)	
    return params
'''
def set_weights(params, new_weights):
    offset = 0
    for module, name, param in params:
        shape = param.shape
        size = torch.prod(torch.tensor(shape)).item()
        value = new_weights[offset:offset + size].view(shape)

        setattr(module, name, value.view(shape))
        offset += size
'''

def set_weights(params, w):	
    offset = 0
    for module, name, shape in params:
        size = np.prod(shape)	       
        value = w[offset:offset + size]
        setattr(module, name, value.view(shape))	
        offset += size

# Example usage
net = LeNet()
weights = torch.load('checkpoints/LeNet5_acc_95.12%.pth')
net.load_state_dict(weights)

theta_map = torch.cat([w.flatten() for w in weights.values()])


params = extract_parameters(net)

def loss_function(output, target):
    return torch.nn.functional.cross_entropy(output, target)


m = torch.nn.Parameter(torch.ones_like(theta_map))
v = torch.nn.Parameter(torch.ones_like(theta_map))

new_weights = 2*m+3*v

optimizer = optim.Adam(params=[m, v], lr=0.1)

set_weights(params, new_weights)
params1 = extract_parameters(net)


optimizer.zero_grad()
output = net(xtrain)
loss = loss_function(output, ytrain)
loss.backward()
optimizer.step()
print("m.grad", m.grad)
print("v.grad", v.grad)

# Ensure the script runs without errors
print("Weights updated successfully.")