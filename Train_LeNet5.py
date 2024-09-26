from torchvision.datasets import MNIST, FashionMNIST
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

class LeNet(nn.Module):

    # network structure
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        '''
        One forward pass through the network.
        
        Args:
            x: input
        '''
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        '''
        Get the number of features in a batch of tensors `x`.
        '''
        size = x.size()[1:]
        return np.prod(size)

def backprop_deep(xtrain, ltrain, net, epochs, B=100, lr=.001, weight_decay=0):
    '''
    Backprop.
    
    Args:
        xtrain: training samples
        ltrain: testing samples
        net: neural network
        epochs: number of epochs
        B: minibatch size
        gamma: step size
        rho: momentum
    '''
    N = xtrain.size()[0]     # Training set size
    NB = N//B                # Number of minibatches
    print(f'Number of minibatches: {NB}. Number of data points: {N}.')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch in range(epochs):
        running_loss = 0.0
        shuffled_indices = np.random.permutation(NB)
        for k in range(NB):
            # Extract k-th minibatch from xtrain and ltrain
            minibatch_indices = range(shuffled_indices[k]*B, (shuffled_indices[k]+1)*B)
            inputs = xtrain[minibatch_indices]
            labels = ltrain[minibatch_indices]

            # Initialize the gradients to zero
            optimizer.zero_grad()

            # Forward propagation
            outputs = net(inputs)

            # Error evaluation
            loss = criterion(outputs, labels)

            # Back propagation
            loss.backward()

            # Parameter update
            optimizer.step()

            # Print averaged loss per minibatch every 100 mini-batches
            # Compute and print statistics
            with torch.no_grad():
                running_loss += loss.item()
            if k % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, k + 1, running_loss / 100))
                running_loss = 0.0


# load the data
xtrain, ytrain = torch.load('./datasets/fashion_mnist_train.pt')
xtest, ytest = torch.load('./datasets/fashion_mnist_test.pt')

net = LeNet()
print('LeNet model architechture: ', net)

start = time.time()
backprop_deep(xtrain, ytrain, net, epochs=50, B = 200, lr = 0.001, weight_decay = 0.0005)
end = time.time()
print(f'It takes {end-start:.6f} seconds.')

# Test the network
y = net(xtest)
test_acc = 100 * (ytest==y.max(1)[1]).float().mean()
print(f'Test accuracy: {test_acc:.2f}%')

# Save the model
torch.save(net.state_dict(), f'LeNet5_Mnist_acc_{test_acc:.2f}%.pth')

#visualize 10 predictions in 2x5 grid with corresponding true labels
fig, axs = plt.subplots(2, 5, figsize=(10, 5))
for i in range(10):
    ax = axs[i//5, i%5]
    ax.imshow(xtest[i].squeeze().numpy(), cmap='gray')
    ax.set_title(f'Predicted: {y[i].argmax().item()}\nTrue: {ytest[i].item()}')
    ax.axis('off')

#plt.savefig('predictions_Lenet5.png')

        




