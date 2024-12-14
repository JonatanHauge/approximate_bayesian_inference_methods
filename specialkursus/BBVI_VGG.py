from utils import BlackBoxVariationalInference, log_like_NN_classification
import numpy as np
import torch 
import sys
from models.VGG16 import VGG16
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms

# settings
sys.stdout.flush()
print('BBVI job is now running')
dataset = 'cifar100'
seed = 4242
torch.manual_seed(seed)
max_itt = 50000
start_plots = [0, 20000, 40000]
step_size = 0.0005
batch_size = 1000
T = 50
K = 100
prior_sigma = 0.1
verbose = True
save_fig = True

fashion_MLE_log_like = -7202.52
mnist_MLE_log_like = -534.40

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load data
if dataset == 'cifar100':
    mean = [0.507, 0.4865, 0.4409]
    std = [0.2673, 0.2564, 0.2761]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_data = CIFAR100(root='./datasets', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = CIFAR100(root='./datasets', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=False)
    weights = torch.load('./checkpoints/cifar100_vgg16_bn-7d8c4031.pt')
        
assert len(test_loader.dataset) % test_loader.batch_size == 0, 'Test loader batch size must divide test set size'

VGG16.to(device)
# Assuming `weights` is your dictionary containing the model weights
theta_map = torch.cat(
    [w.flatten() for name, w in weights.items() if not any(substring in name for substring in ['running_mean', 'running_var', 'num_batches_tracked'])]
)
theta_map = theta_map.to(device=device)

# Print the keys of the weights that were included
print("Filtered Weight keys: ", [name for name in weights.keys() if not any(substring in name for substring in ['running_mean', 'running_var', 'num_batches_tracked'])])


num_params = sum(p.numel() for p in VGG16.parameters())
P = torch.randn(num_params, K, device=device)
P /= torch.norm(P, dim=0)  # Normalize columns to norm 1

#net.load_state_dict(weights)
#fashion_MLE_log_like = -torch.nn.CrossEntropyLoss(reduction='sum')(net(xtrain), ytrain).item()
#print(fashion_MLE_log_like)

bbvi = BlackBoxVariationalInference(VGG16, theta_map, P, log_like_NN_classification, 
                                    K, step_size, max_itt, batch_size, seed, verbose, 
                                    T, prior_sigma, device)

bbvi.fit(train_loader)

acc, entropy, lpd, ece, mce = bbvi.compute_all_metrics(test_loader, num_samples=100, num_bins=10)

print('Accuracy:', np.round(acc, 3))
print('Entropy:', np.round(entropy, 3))
print('LPD:', np.round(lpd, 3))
print('ECE:', np.round(ece, 3))
print('MCE:', np.round(mce, 3))

for start_plot in start_plots:
    fig, axes = plt.subplots(3, 2, figsize=(10, 6))


    fig.suptitle(f'{dataset}_Theta_MAP plot from {start_plot} with K={K}, T={T}, batch_size={batch_size}, step_size={step_size}\nAccuracy: {acc:.3f}, Entropy: {entropy:.3f}, ECE: {ece:.3f}, MCE={mce:.3f}, LPD: {lpd:.3f}, sigma_init=exp(-1)')

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
    #axes[2,1].set_ylim(bottom = -4, top = -2)
    axes[2,1].set_ylabel('log(v)')

    number_of_param_to_plot = min(K,50)

    for i in range(2,number_of_param_to_plot):
        axes[1,1].plot(range(start_plot, max_itt),bbvi.m_history[start_plot:, i], '-', linewidth=0.5)
        axes[2,1].plot(range(start_plot, max_itt), np.log(bbvi.v_history[start_plot:, i]), '-', linewidth=0.5)

    for i in range(6):
        axes.flat[i].legend(loc = 'lower left')

    if save_fig:
        plt.savefig(f'plots/{dataset}_Theta_MAP_plotfrom={start_plot}_K={K}_T={T}_batch_size={batch_size}_step_size={step_size}_acc={acc:.3}_Entropy={entropy:.3f}_ECE={ece:.3f}_LPD={lpd:.3f}_MCE={mce:.3f}_sigma_init=exp(-1).png')

    plt.show()