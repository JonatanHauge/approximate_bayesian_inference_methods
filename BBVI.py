from utils import BlackBoxVariationalInference, log_like_NN_classification
import numpy as np
import torch 
from models.LeNet5 import LeNet
import pandas as pd

# settings
dataset = 'fashion_mnist'
seed = 4242
torch.manual_seed(seed)
max_itt = 20000
start_plot = 0
step_size = 0.0005
batch_size = 500
T = 2000
K = 5
prior_sigma = 0.5
random = False
verbose = True
save_fig = True


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load data
if dataset == 'mnist':
    xtrain, ytrain = torch.load('./datasets/fashion_mnist_train.pt', map_location=device)
    xtest, ytest = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
    weights = torch.load('./checkpoints/LeNet5_FashionMnist_acc_90.55%.pth') # load the weights
elif dataset == 'fashion_mnist':
    xtrain, ytrain = torch.load('./datasets/fashion_mnist_train.pt', map_location=device)
    xtest, ytest = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
    weights = torch.load('./checkpoints/LeNet5_Mnist_acc_98.94%.pth') # load the weights

net = LeNet().to(device=device) 
theta_map = torch.cat([w.flatten() for w in weights.values()]) # flatten the weights
if random:
    theta_map = torch.zeros_like(theta_map)
theta_map = theta_map.to(device=device)

num_params = sum(p.numel() for p in net.parameters())
P = torch.randn(num_params, K, device=device)
P /= torch.norm(P, dim=0)  # Normalize columns to norm 1

bbvi = BlackBoxVariationalInference(net, theta_map, P, log_like_NN_classification, 
                                    K, step_size, max_itt, batch_size, seed, verbose, 
                                    T, prior_sigma, device)

bbvi.fit(xtrain, ytrain)

acc = bbvi.compute_accuracy(xtest, ytest, num_samples=100) #calculate accuracy
entropy = bbvi.compute_entropy_posterior_predictive(xtest)
lpd = bbvi.compute_LPD(xtest, ytest)
ece = bbvi.compute_ECE(xtest, ytest)

print('Accuracy:', np.round(acc, 3))
print('Entropy:', np.round(entropy, 3))
print('LPD:', np.round(lpd, 3))
print('ECE:', np.round(ece, 3))


import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 2, figsize=(10, 6))

if random:
    fig.suptitle(f'Zeros_theta with K={K}, T={T}, batch_size={batch_size}, step_size={step_size}\nAccuracy: {acc:.3f}, Entropy: {entropy:.3f}, ECE: {ece:.3f}, LPD: {lpd:.3f}')
else:
    fig.suptitle(f'Theta_MAP with K={K}, T={T}, batch_size={batch_size}, step_size={step_size}\nAccuracy: {acc:.3f}, Entropy: {entropy:.3f}, ECE: {ece:.3f}, LPD: {lpd:.3f}')



log_like_series = pd.Series(bbvi.log_like_history)
running_mean_log_like = log_like_series.rolling(window=500).mean().dropna()

ELBO_series = pd.Series(bbvi.ELBO_history)
running_mean_ELBO = ELBO_series.rolling(window=500).mean().dropna()

log_prior_series = pd.Series(bbvi.log_prior_history)
running_mean_log_prior = log_prior_series.rolling(window=500).mean().dropna()


axes[0,0].plot(running_mean_log_like, label=f'log-likelihood/{T}')
axes[1,0].plot(running_mean_log_prior, label='log-prior')
axes[2,0].plot(range(start_plot, max_itt), bbvi.entropy_history[start_plot:], label='entropy')

axes[0,1].plot(running_mean_ELBO, label='ELBO')
axes[1,1].plot(range(start_plot, max_itt), bbvi.m_history[start_plot:, 0], '-', linewidth=0.5, label = "m")
axes[2,1].plot(range(start_plot, max_itt), np.log(bbvi.v_history[start_plot:, 0]), '-', linewidth=0.5, label = "v")
axes[2,1].axhline(y = np.log(prior_sigma), xmin = start_plot, xmax = max_itt)

number_of_param_to_plot = min(K,50)

for i in range(2,number_of_param_to_plot):
    axes[1,1].plot(range(start_plot, max_itt),bbvi.m_history[start_plot:, i], '-', linewidth=0.5)
    axes[2,1].plot(range(start_plot, max_itt),bbvi.v_history[start_plot:, i], '-', linewidth=0.5)

for i in range(6):
    axes.flat[i].legend(loc = 'lower right')

if save_fig:
    if random:
        plt.savefig(f'plots/{dataset}_Zeros_theta_K={K}_T={T}_batch_size={batch_size}_step_size={step_size}_acc={acc:.3}_Entropy={entropy:.3f}_ECE={ece:.3f}=LPD: {lpd:.3f}.png')
    else:
        plt.savefig(f'plots/{dataset}Theta_MAP_K={K}_T={T}_batch_size={batch_size}_step_size={step_size}_acc={acc:.3}_Entropy={entropy:.3f}_ECE={ece:.3f}=LPD: {lpd:.3f}.png')

plt.show()