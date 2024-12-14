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
    
    def __init__(self, model, theta_init, P, intrinsic_dim=5, lr=1e-2, batch_size=20, weight_decay=0.0005, epochs=10, verbose=False, seed = 0, name='Train_intrinsic', device=None):
        
        self.name = name
        self.params = extract_parameters(model)
        self.model = model.to(device)  # Move model to device

        self.verbose = verbose        
        self.X, self.y, self.loss = None, None, None
        self.intrinsic_dim, self.lr, self.epochs, self.weight_decay, self.seed = num_params, lr, epochs, weight_decay, seed
        self.batch_size = batch_size
        self.theta_init, self.P = theta_init.to(device), P.to(device)  # Move theta_init and P to device
        self.device = device  # Store device

        
        # Initialize the variational parameters for the mean-field approximation
        self.z = nn.Parameter(torch.ones(intrinsic_dim, device=device), requires_grad=True)  # z on device 
        # prepare optimizer
        self.optimizer = optim.Adam(params=[self.z], lr=self.lr, weight_decay=self.weight_decay)

    def train(self, X, y):
        """ fits the variational approximation q given data (X,y) by maximizing the ELBO using gradient-based methods """ 
        torch.manual_seed(self.seed)
        self.X, self.y, N = X.to(self.device), y.to(self.device), len(X)
        NB = N//self.batch_size                # Number of minibatches
        self.loss_history, self.z_history = [], []

        print('Start training')      
        t0 = time()
        for e in range(self.epochs):
            if e > self.epochs - 5:
                new_weight_decay = 0.0005
                self.optimizer = optim.Adam(params=[self.z], lr=self.lr, weight_decay=new_weight_decay)

            running_loss = 0.0
            shuffled_indices = np.random.permutation(NB)
            for k in range(NB):
                # Extract k-th minibatch from xtrain and ltrain
                minibatch_indices = range(shuffled_indices[k]*self.batch_size, (shuffled_indices[k]+1)*self.batch_size)
                inputs = self.X[minibatch_indices].to(self.device)
                labels = self.y[minibatch_indices].to(self.device)
                
                # evaluate loss
                loss = loss_function(self.model, self.params, inputs, labels, self.z, self.P, self.theta_init)
                    
                # store current values for plotting purposes
                self.loss_history.append(loss.clone().detach().cpu().numpy())  # Move to CPU before appending
                self.z_history.append(self.z.clone().detach().cpu().numpy())  # Move to CPU before appending
        
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
            X = X.to(self.device)  # Move inputs to the correct device
            out = torch.softmax(self.model(X), dim=1)  # Apply softmax to get probabilities
        return out.argmax(dim=1)
        
    def compute_accuracy(self, X, y):
        y_pred = self.predict(X)
        return (y_pred == y.to(self.device)).float().mean().item()


def loss_function(model, params, X, y, z, P, theta_init):
    theta = z @ P.T + theta_init
    set_weights(params, theta)
    #outputs = model(X)
    nll = torch.nn.CrossEntropyLoss(reduction='sum')(model(X), y)
    return nll


# Load data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xtrain, ytrain = torch.load('./datasets/mnist_train.pt', map_location=device)
xtest, ytest = torch.load('./datasets/mnist_test.pt', map_location=device)

print('xtrain:', xtrain.shape, xtrain.dtype)
print('ytrain:', ytrain.shape, ytrain.dtype)

K_list = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 500] # List of intrinsic dimensions to try
#K_list = [5]
acc_list = [] # List to store the test accuracy for each K


seed = 0
save_fig = True
torch.manual_seed(seed)
theta_MAP = torch.cat([w.flatten() for w in torch.load('checkpoints\LeNet5_acc_95.12%.pth').values()])
theta_random = torch.zeros_like(theta_MAP)
print(f'Norm difference between theta_MAP and theta_random: {torch.norm(theta_MAP - theta_random)}')

for K in K_list:
    #Settings
    print(f"running for K={K}")
    lr = 0.1
    epochs = 15
    B = 100
    weight_decay = 0
    save_model = True

    net = LeNet().to(device)
    num_params = sum(p.numel() for p in net.parameters())
    P = torch.randn(num_params, K).to(device)
    P /= torch.norm(P, dim=0)  # Normalize columns to norm 1

    Train_obj = Train_intrinsic(net, theta_random, P, K, lr=lr, batch_size=B, weight_decay=weight_decay, epochs=epochs, verbose=True, device=device)

    #Start training
    start = time()
    Train_obj.train(xtrain, ytrain)
    end = time()
    print(f'It takes {end-start:.6f} seconds to train for {epochs} epochs.')

    test_acc = Train_obj.compute_accuracy(xtest, ytest)

    print(f'Test accuracy: {100*test_acc:.2f}%')
    acc_list.append(test_acc)

# Save the model
    if save_model:
        torch.save(net.state_dict(), f'checkpoints/LeNet5_K_{K}_acc_{test_acc:.2f}%.pth')


if save_fig:
    np.save(f'checkpoints/acc_list/LeNet5_acc_list.npy', np.array(acc_list),)
    fig, ax = plt.subplots()
    ax.plot(K_list, acc_list, '-o')
    ax.set_xlabel('Subspace dimension K')
    ax.set_ylabel('Test accuracy')
    ax.set_title('Test accuracy vs. Intrinsic dimension K')
    plt.savefig(f'plots/Test_accuracy_vs_K_lr_{lr}_epohcs_{epochs}_weight_decay_{weight_decay}.png')
    plt.show()
