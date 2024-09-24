from torchvision.datasets import MNIST
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from time import time
from utils import extract_parameters, set_weights
from models.LeNet5 import LeNet

softmax = lambda x: torch.exp(x - torch.max(x, dim=1, keepdim=True)[0]) / torch.sum(torch.exp(x - torch.max(x, dim=1, keepdim=True)[0]), dim=1, keepdim=True)

class Train_intrinsic(object):
    
    def __init__(self, model, theta_init, P, intrinsic_dim=5, lr=1e-2, batch_size=20, epochs=10, verbose=False, seed = 0, name='Train_intrinsic'):
        
        self.name = name
        self.params = extract_parameters(model)
        self.model = model

        self.verbose = verbose        
        self.X, self.y, self.loss = None, None, None
        self.intrinsic_dim, self.lr, self.epochs, self.seed = num_params, lr, epochs, seed
        self.batch_size = batch_size
        self.theta_init, self.P = theta_init, P
        
        # Initialize the variational parameters for the mean-field approximation
        self.z = nn.Parameter(torch.ones(intrinsic_dim), requires_grad=True)  
        # prepare optimizer
        self.optimizer = optim.Adam(params=[self.z], lr=self.lr)

    def train(self, X, y):
        """ fits the variational approximation q given data (X,y) by maximizing the ELBO using gradient-based methods """ 
        torch.manual_seed(self.seed)
        self.X, self.y, N = X, y, len(X)
        NB = N//self.batch_size                # Number of minibatches
        self.loss_history, self.z_history = [], []

        print('Start training')      
        t0 = time()
        for e in range(self.epochs):

            running_loss = 0.0
            shuffled_indices = np.random.permutation(NB)
            for k in range(NB):
                # Extract k-th minibatch from xtrain and ltrain
                minibatch_indices = range(shuffled_indices[k]*self.batch_size, (shuffled_indices[k]+1)*self.batch_size)
                inputs = self.X[minibatch_indices]
                labels = self.y[minibatch_indices]
                
                # evaluate loss
                loss = loss_function(self.model, self.params, inputs, labels, self.z, self.P, self.theta_init)
                    
                # store current values for plotting purposes
                self.loss_history.append(loss.clone().detach().numpy())
                self.z_history.append(self.z.clone().detach().numpy())
        
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    running_loss += loss.item()

            # verbose?
            if self.verbose:
                print(f'Epoch {e+1}/{self.epochs}, mean loss per batch: {running_loss/NB:.3f}')
            running_loss = 0.0

    def predict(self, X):
        with torch.no_grad():
            out = softmax(self.model(X))
        return out.argmax(dim=1)
        
    def compute_accuracy(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y).float().mean().item()


def loss_function(model, params, X, y, z, P, theta_init):
    theta = z @ P.T + theta_init
    set_weights(params, theta)
    outputs = model(X)
    nll = torch.nn.CrossEntropyLoss(reduction='sum')(model(X), y)
    return nll


# Load data
xtrain, ytrain = torch.load('./datasets/mnist_train.pt')
xtest, ytest = torch.load('./datasets/mnist_test.pt')

print('xtrain:', xtrain.shape, xtrain.dtype)
print('ytrain:', ytrain.shape, ytrain.dtype)

#Settings
K = 200
lr = 0.1
epochs = 10
B = 100
weight_decay = 0
save_model = False
save_fig = False
seed = 0

net = LeNet()
torch.manual_seed(seed)
num_params = sum(p.numel() for p in net.parameters())
P = torch.randn(num_params, K)
P /= torch.norm(P, dim=0)  # Normalize columns to norm 1
theta_MAP = torch.cat([w.flatten() for w in torch.load('checkpoints\LeNet5_acc_95.12%.pth').values()])
theta_random = torch.zeros_like(theta_MAP)
print(f'Norm difference between theta_MAP and theta_random: {torch.norm(theta_MAP - theta_random)}')

Train_obj = Train_intrinsic(net, theta_random, P, K, lr=lr, batch_size=B, epochs=epochs, verbose=True)

#Start training
start = time()
#backprop_deep(xtrain, ytrain, net, K, P, theta_random, epochs=5)
Train_obj.train(xtrain, ytrain)
end = time()
print(f'It takes {end-start:.6f} seconds to train for {epochs} epochs.')

test_acc = Train_obj.compute_accuracy(xtest, ytest)

print(f'Test accuracy: {100*test_acc:.2f}%')

# Save the model
if save_model:
    torch.save(net.state_dict(), f'LeNet5_K_{K}_acc_{test_acc:.2f}%.pth')

#visualize 10 predictions in 2x5 grid with corresponding true labels
fig, axs = plt.subplots(2, 5, figsize=(10, 5))
#for i in range(10):
##    ax = axs[i//5, i%5]
 #   ax.imshow(xtest[i].squeeze().numpy(), cmap='gray')
  #  ax.set_title(f'Predicted: {y[i].argmax().item()}\nTrue: {ytest[i].item()}')
  #  ax.axis('off')
#plt.show()

if save_fig:
    plt.savefig('predictions_Lenet5.png')

        

    
def backprop_deep(xtrain, ltrain, net, K, P, theta_init, epochs, B=100, gamma=.001, weight_decay=.005):
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
        epsilon = torch.randn(self.num_params)
            z_sample = self.m + torch.sqrt(torch.exp(self.v)) * epsilon  # shape:  (,K)
            w_sample = z_sample @ self.P.T + self.theta_map
    '''
    N = xtrain.size()[0]     # Training set size
    NB = N//B                # Number of minibatches

    z = nn.Parameter(torch.ones(K), requires_grad=True)
    w = z @ P.T + theta_init
    params = extract_parameters(net)
    optimizer = torch.optim.Adam([z], lr=gamma, weight_decay=weight_decay)
    
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
            
            loss = criterion(net, w, params, inputs, labels)
            loss.backward()
            optimizer.step()
            
            with torch.no_grad(): # Compute and print statistics
                running_loss += loss.item()
                if k % 100 == 99:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, k + 1, running_loss / 100))
                    running_loss = 0.0


