#%%
from utils import VariationalInference, BlackBoxVariationalInference, log_prior_pdf, log_like_NN_classification
import numpy as np
import torch 
from models.LeNet5 import LeNet

# Load data
xtrain, ytrain = torch.load('./datasets/mnist_train.pt')
xtest, ytest = torch.load('./datasets/mnist_test.pt')

print('xtrain:', xtrain.shape, xtrain.dtype)
print('ytrain:', ytrain.shape, ytrain.dtype)
#%%

net = LeNet()
weights = torch.load('checkpoints\LeNet5_acc_95.12%.pth') # load the weights
theta_map = torch.cat([w.flatten() for w in weights.values()]) # flatten the weights


# settings
seed = 4242
torch.manual_seed(seed)
max_itt = 4000
step_size = 0.01
batch_size = 100
T = 5000
K = 100
random = True
if random:
    theta_map = torch.randn_like(theta_map)
verbose = True
save_fig = True


num_params = sum(p.numel() for p in net.parameters())
P = torch.randn(num_params, K)
P /= torch.norm(P, dim=0)  # Normalize columns to norm 1
bbvi = BlackBoxVariationalInference(net, theta_map, P, log_prior_pdf, log_like_NN_classification, 
                                    K, step_size, max_itt, batch_size, seed, verbose, T)

bbvi.fit(xtrain, ytrain)

#calculate accuracy
acc = bbvi.compute_accuracy(xtest, ytest, num_samples=50)
print('Accuracy:', np.round(acc, 3))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 2, figsize=(10, 6))

if random:
    fig.suptitle(f'Random theta with K={K}, T={T}, batch_size={batch_size}, step_size={step_size}\nAccuracy: {acc:.3f}')
else:
    fig.suptitle(f'Theta_MAP with K={K}, T={T}, batch_size={batch_size}, step_size={step_size}\nAccuracy: {acc:.3f}')

axes[0,0].plot(bbvi.log_like_history, label=f'log-likelihood/{T}')
axes[1,0].plot(bbvi.log_prior_history, label='log-prior')
axes[2,0].plot(bbvi.entropy_history, label='entropy')

axes[0,1].plot(bbvi.ELBO_history, label='ELBO')

for i in range(5):
    axes[1,1].plot(range(max_itt), bbvi.m_history[:, i], '-', linewidth=0.5)
    axes[2,1].plot(range(max_itt), bbvi.v_history[:, i], '-', linewidth=0.5)

for i in range(6):
    axes.flat[i].legend()

if save_fig:
    if random:
        plt.savefig(f'plots/Random_theta_K={K}_T={T}_batch_size={batch_size}_step_size={step_size}_acc={acc:.3}.png')
    else:
        plt.savefig(f'plots/Theta_MAP_K={K}_T={T}_batch_size={batch_size}_step_size={step_size}_acc={acc:.3}.png')

plt.show()