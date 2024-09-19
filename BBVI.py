from utils import BlackBoxVariationalInference
import numpy as np
import torch 
from torchvision.datasets import MNIST
from models.LeNet5 import LeNet

# load the MNIST dataset
mnist_train = MNIST('./datasets', train=True, download=False)
mnist_test = MNIST('./datasets', train=False, download=False)

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



def set_weights(params, w):	
    offset = 0
    for module, name, shape in params:
        size = np.prod(shape)	       
        value = w[offset:offset + size]
        setattr(module, name, value.view(shape))	
        offset += size

def log_prior_pdf(z, prior_mean=torch.tensor(0), prior_var=torch.tensor(1)):
    """ Evaluates the log prior Gaussian for each sample of z. 
        D denote the dimensionality of the model and S denotes the number of MC samples.

        Inputs:
            z             -- np.array of shape (S, 2*K)
            prior_mean    -- np.array of shape (S, K)
            prior_var     -- np.array of shape (S, K)

        Returns:
            log_prior     -- np.array of shape (1,)???
       """
    log_prior = torch.sum(log_npdf(z, prior_mean, prior_var))
    return log_prior

def log_like_NN_classification(X, y, theta):
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

    log_likelihood = torch.tensor(0.0, device=theta.device)  # Initialize as a tensor on the same device as theta_s
    params = extract_parameters(net)   
    set_weights(params, theta)
    outputs = softmax(net(X)) # Forward pass
    log_probs = torch.log(outputs + 1e-6)  # Adding epsilon to avoid log(0)
    log_likelihood += torch.sum(log_probs[range(len(y)), y])
    
    return log_likelihood


net = LeNet()
#load weights
weights = torch.load('checkpoints\LeNet5_acc_95.12%.pth')
theta_map = torch.cat([w.flatten() for w in weights.values()]) # flatten the weights
 # set requires_grad to False??
theta_map = theta_map.detach().requires_grad_(True)

#theta_map = torch.zeros_like(theta_map)

# settings
num_params = sum(p.numel() for p in net.parameters())
print('Number of parameters:', num_params)
K = 5
P = torch.randn(num_params, K)
P /= torch.norm(P, dim=0)  # Normalize columns to norm 1
max_itt = 2000
step_size = 0.1
batch_size = 100
T = 1
seed = 1
verbose = True

bbvi = BlackBoxVariationalInference(theta_map, P, log_prior_pdf, log_like_NN_classification, 
                                    K, step_size, max_itt, batch_size, seed, verbose, T)
bbvi.fit(xtrain, ytrain)


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].plot(bbvi.ELBO_history, label='BBVI')
axes[0].set(title='Evidence lower bound', xlabel='Iterations')
axes[0].legend()

for i in range(5):
    axes[1].plot(range(max_itt), bbvi.m_history[:, i], '-', linewidth=0.5)
    axes[2].plot(range(max_itt), bbvi.v_history[:, i], '-', linewidth=0.5)

axes[2].set(xlabel='Posterior variance $v_1$', ylabel='Posterior variance $v_2$', title='Optimization trajectory for variances')

for i in range(3):
    axes.flat[i].legend()

plt.show()