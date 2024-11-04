from utils_flows import BBVI_NormalizingFlows, log_like_NN_classification
import numpy as np
import torch 
import sys
from models.LeNet5 import LeNet
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# settings
sys.stdout.flush()
print('BBVI job is now running')
dataset = 'fashion_mnist'
seed = 4242
torch.manual_seed(seed)
max_itt = 1000
start_plots = [0]
step_size = 0.0005
batch_size = 500
T = 2500
K = 10
prior_sigma = 0.1
SWA = True
verbose = True
save_fig = True

num_transformations = 3

fashion_MLE_log_like = -7202.52
mnist_MLE_log_like = -534.40

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load data
if dataset == 'mnist':
    xtrain, ytrain = torch.load('./datasets/mnist_train.pt', map_location=device)
    train_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True)
    xtest, ytest = torch.load('./datasets/mnist_test.pt', map_location=device)
    test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, shuffle=False)
    weights = torch.load('./checkpoints/LeNet5_Mnist_acc_98.94_.pth') # load the weights
elif dataset == 'fashion_mnist':
    xtrain, ytrain = torch.load('./datasets/fashion_mnist_train.pt', map_location=device)
    train_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True)
    xtest, ytest = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
    test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, shuffle=False)
    weights = torch.load('./checkpoints/LeNet5_FashionMnist_acc_90.55_.pth')


assert len(test_loader.dataset) % test_loader.batch_size == 0, 'Test loader batch size must divide test set size'

net = LeNet().to(device=device) 
theta_map = torch.cat([w.flatten() for w in weights.values()]) # flatten the weights

theta_map = theta_map.to(device=device)

num_params = sum(p.numel() for p in net.parameters())
P = torch.randn(num_params, K, device=device)
P /= torch.norm(P, dim=0)  # Normalize columns to norm 1


bbvi = BBVI_NormalizingFlows(net, theta_map, P, log_like_NN_classification, 
                                    K, step_size, max_itt, batch_size, seed, verbose, 
                                    T, prior_sigma, SWA, device, num_transformations)

bbvi.fit(train_loader)

print(bbvi.parameters)

acc, entropy, lpd, ece, mce = bbvi.compute_all_metrics(test_loader, num_samples=100, num_bins=10)

#OOD error 
if dataset == 'mnist':
    xood, yood = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
    ood_loader = DataLoader(TensorDataset(xood, yood), batch_size=batch_size, shuffle=False)
    ood = bbvi.compute_entropy_posterior_predictive(ood_loader)
elif dataset == 'fashion_mnist':
    xood, yood = torch.load('./datasets/mnist_test.pt', map_location=device)
    ood_loader = DataLoader(TensorDataset(xood, yood), batch_size=batch_size, shuffle=False)
    ood = bbvi.compute_entropy_posterior_predictive(ood_loader)

# print results
print('Accuracy:', np.round(acc, 3))
print('Entropy:', np.round(entropy, 3))
print('LPD:', np.round(lpd, 3))
print('ECE:', np.round(ece, 3))
print('MCE:', np.round(mce, 3))
print('OOD:', np.round(ood, 3))


for start_plot in start_plots:
    fig, axes = plt.subplots(3, 2, figsize=(10, 6))
  
    fig.suptitle(f'{dataset}_Theta_MAP plot from {start_plot} with K={K}, T={T}, batch_size={batch_size}, step_size={step_size}\nAccuracy: {acc:.3f}, Entropy: {entropy:.3f}, ECE: {ece:.3f}, MCE={mce:.3f}, LPD: {lpd:.3f}, OOD: {ood:.3f}, sigma_init=exp(-1)')

    log_like_series = pd.Series(bbvi.log_like_history)
    running_mean_log_like = log_like_series.rolling(window=500).mean().dropna()

    ELBO_series = pd.Series(bbvi.ELBO_history)
    running_mean_ELBO = ELBO_series.rolling(window=500).mean().dropna()

    log_prior_series = pd.Series(bbvi.log_prior_history)
    running_mean_log_prior = log_prior_series.rolling(window=500).mean().dropna()

    axes[0,0].plot(range(start_plot, start_plot + len(running_mean_log_like[start_plot:])), running_mean_log_like[start_plot:], label=f'log-likelihood/{T}')
    if dataset == 'mnist':
        axes[0,0].axhline(y = mnist_MLE_log_like/T, color = 'r', label = f'MLE_log_like/{T}', linestyle = '--')
        axes[0,0].set_ylim(bottom = np.min(running_mean_log_like[start_plot:])*1.1, top = mnist_MLE_log_like/T* 0.9)
    elif dataset == 'fashion_mnist':
        axes[0,0].axhline(y = fashion_MLE_log_like/T, color = 'r', label = f'MLE_log_like/{T}', linestyle = '--')
        axes[0,0].set_ylim(np.min(running_mean_log_like[start_plot:])*1.1, fashion_MLE_log_like/T*0.9)

    axes[1,0].plot(range(start_plot, start_plot + len(running_mean_log_like[start_plot:])), running_mean_log_prior[start_plot:], label='log-prior')
    axes[2,0].plot(range(start_plot, max_itt), bbvi.entropy_history[start_plot:], label='entropy')

    axes[0,1].plot(range(start_plot, start_plot + len(running_mean_log_like[start_plot:])), running_mean_ELBO[start_plot:], label='ELBO')
    axes[1,1].plot(range(start_plot, max_itt), bbvi.m_history[start_plot:, 0], '-', linewidth=0.5, label = "prior m")
    axes[2,1].plot(range(start_plot, max_itt), np.log(bbvi.v_history[start_plot:, 0]), '-', linewidth=0.5, label = "log(prior v)")
    axes[2,1].axhline(y = np.log(prior_sigma))
    axes[2,1].set_ylim(bottom = -4, top = -2)
    axes[2,1].set_ylabel('log(v)')

    number_of_param_to_plot = min(K,50)

    for i in range(2,number_of_param_to_plot):
        axes[1,1].plot(range(start_plot, max_itt),bbvi.m_history[start_plot:, i], '-', linewidth=0.5)
        axes[2,1].plot(range(start_plot, max_itt), np.log(bbvi.v_history[start_plot:, i]), '-', linewidth=0.5)

    for i in range(6):
        axes.flat[i].legend(loc = 'lower left')

    if save_fig:
        
        plt.savefig(f'plots/{dataset}_Theta_MAP_plotfrom={start_plot}_K={K}_T={T}_batch_size={batch_size}_step_size={step_size}_acc={acc:.3}_Entropy={entropy:.3f}_ECE={ece:.3f}_LPD={lpd:.3f}_OOD={ood:.2f}_MCE={mce:.3f}_sigma_init=exp(-1).png')

    plt.show()