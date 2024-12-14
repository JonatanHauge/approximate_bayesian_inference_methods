import torch
from torchvision import datasets, transforms

# Load and convert to tensors
mnist_train = datasets.MNIST(root='./datasets', train=True, download=True)
mnist_test = datasets.MNIST(root='./datasets', train=False, download=True)

train_data = mnist_train.data
train_targets = mnist_train.targets
test_data = mnist_test.data
test_targets = mnist_test.targets

# Normalize the data
train_data = train_data.float()/255
test_data = test_data.float()/255

# Insert a channel dimension
train_data = train_data.unsqueeze(1)
test_data = test_data.unsqueeze(1)

# Save tensors to disk (do this once)
torch.save((train_data, train_targets), './datasets/mnist_train.pt')
torch.save((test_data, test_targets), './datasets/mnist_test.pt')