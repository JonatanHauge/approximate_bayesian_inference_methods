from utils import VariationalInference, BlackBoxVariationalInference
import numpy as np
import torch 
from torchvision.datasets import MNIST
from models.LeNet5 import LeNet

# load the MNIST dataset
mnist_train = MNIST('./datasets', train=True, download=True)
mnist_test = MNIST('./datasets', train=False, download=True)

# load the data
xtrain = mnist_train.train_data
ytrain = mnist_train.train_labels
xtest = mnist_test.test_data
ytest = mnist_test.test_labels

# normalize the data
xtrain = xtrain.float()/255
xtest = xtest.float()/255

#insert a channel dimension
xtrain = xtrain.unsqueeze(1)
xtest = xtest.unsqueeze(1)

log_npdf = lambda x, m, v: -(x - m) ** 2 / (2 * v) - 0.5 * torch.log(2 * torch.pi * v)
log_mvnpdf = lambda x, m, v: -0.5 * torch.sum((x - m) ** 2 / v + torch.log(2 * torch.pi * v), dim=1)
#softmax = lambda x: np.exp(x) / np.sum(np.exp(x), axis=1)[:, None]
softmax = lambda x: torch.exp(x - torch.max(x, dim=1, keepdim=True)[0]) / torch.sum(torch.exp(x - torch.max(x, dim=1, keepdim=True)[0]), dim=1, keepdim=True)


def new_weights_in_NN(model, new_weight_vector):
    current_index = 0
    # Iterate over each parameter in the model
    for param in model.parameters():
        num_params = param.numel() # number of elements in the tensor
        new_weights = new_weight_vector[current_index:current_index + num_params].view_as(param.data) # reshape the new weights to the shape of the parameter tensor
        param.data.copy_(new_weights) # copy the new weights to the parameter tensor
        current_index += num_params # update the current index

    return model

def set_weights(model, vector):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()))
        offset += param.numel()

def log_prior_pdf(z, prior_mean, prior_var):
    """ Evaluates the log prior Gaussian for each sample of z. 
        D denote the dimensionality of the model and S denotes the number of MC samples.

        Inputs:
            z             -- np.array of shape (S, 2*K)
            prior_mean    -- np.array of shape (S, K)
            prior_var     -- np.array of shape (S, K)

        Returns:
            log_prior     -- np.array of shape (1,)???
       """
    log_prior = torch.sum(log_npdf(z, prior_mean, prior_var), dim=1)
    return log_prior

def log_like_NN_classification(X, y, theta_s):
    """
    Implements the log likelihood function for the classification NN with categorical likelihood.
    S is number of MC samples, N is number of datapoints in likelihood and D is the dimensionality of the model (number of weights).

    Inputs:
    X              -- Data (np.array of size N x D)
    y              -- vector of target (np.array of size N)
    theta_s        -- vector of weights (np.array of size (S, D))

    outputs: 
    log_likelihood -- Array of log likelihood for each sample in z (np.array of size S)
     """
    S = theta_s.size(0)
    log_likelihood = torch.tensor(0.0, device=theta_s.device)  # Initialize as a tensor on the same device as theta_s

    for i in range(S):
        set_weights(net, theta_s[i])# Set the weights for the model
        outputs = softmax(net(X)) # Forward pass
        log_probs = torch.log(outputs + 1e-6)  # Adding epsilon to avoid log(0)
        log_likelihood += torch.sum(log_probs[range(len(y)), y])
    
    return log_likelihood / S


net = LeNet()
#load weights
weights = torch.load('checkpoints\LeNet5_acc_95.12%.pth')
theta_map = torch.cat([w.flatten() for w in weights.values()]) # flatten the weights
 # set requires_grad to False??
theta_map = theta_map.detach().requires_grad_(False)

# settings
num_params = sum(p.numel() for p in net.parameters())
print('Number of parameters:', num_params)
K = 10
P = torch.randn(num_params, K)
P /= torch.norm(P, dim=0)  # Normalize columns to norm 1
max_itt = 10
step_size = 5e-2
num_samples = 2
batch_size = 5
seed = 1
verbose = True

bbvi = BlackBoxVariationalInference(theta_map, P, log_prior_pdf, log_like_NN_classification, K, step_size, max_itt, num_samples, batch_size, seed, verbose)
bbvi.fit(xtrain, ytrain)