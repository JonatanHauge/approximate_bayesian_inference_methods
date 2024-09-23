from utils import BlackBoxVariationalInference, log_like_NN_classification
import numpy as np
import torch 
from models.LeNet5 import LeNet



# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load data
xtrain, ytrain = torch.load('./datasets/mnist_train.pt')
xtest, ytest = torch.load('./datasets/mnist_test.pt')

xtrain = xtrain.to(device=device)
ytrain = ytrain.to(device=device)
xtest = xtest.to(device=device) 
ytest = ytest.to(device=device)


net = LeNet().to(device=device) 
weights = torch.load('./checkpoints/LeNet5_acc_95.12%.pth') # load the weights
theta_map = torch.cat([w.flatten() for w in weights.values()]) # flatten the weights


# settings
seed = 4242
torch.manual_seed(seed)
max_itt = 30000
start_plot = 15000 # when to start plotting (discarding initial iterations for visualization)
step_size = 0.0005
batch_size = 2000
T = 5000
K = 50
prior_sigma = 0.1
random = False
if random:
    theta_map = torch.randn_like(theta_map)

theta_map = theta_map.to(device=device)
verbose = True
save_fig = True


num_params = sum(p.numel() for p in net.parameters())
P = torch.randn(num_params, K, device=device)
P /= torch.norm(P, dim=0)  # Normalize columns to norm 1

bbvi = BlackBoxVariationalInference(net, theta_map, P, log_like_NN_classification, 
                                    K, step_size, max_itt, batch_size, seed, verbose, 
                                    T, prior_sigma, device)

bbvi.fit(xtrain, ytrain)

#calculate accuracy
acc = bbvi.compute_accuracy(xtest, ytest, num_samples=50)
print('Accuracy:', np.round(acc, 3))

entropy = bbvi.compute_entropy_posterior_predictive(xtest)
print('Entropy:', np.round(entropy, 3))

lpd = bbvi.compute_LPD(xtest, ytest)
print('LPD:', np.round(lpd, 3))

ece = bbvi.compute_ECE(xtest, ytest)
print('ECE:', np.round(ece, 3))




import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, 1, figsize=(10, 6))

if random:
    fig.suptitle(f'Random theta with K={K}, T={T}, batch_size={batch_size}, step_size={step_size}\nAccuracy: {acc:.3f}, Entropy: {entropy:.3f}, ECE: {ece:.3f}, LPD: {lpd:.3f}')
else:
    fig.suptitle(f'Theta_MAP with K={K}, T={T}, batch_size={batch_size}, step_size={step_size}\nAccuracy: {acc:.3f}, Entropy: {entropy:.3f}, ECE: {ece:.3f}, LPD: {lpd:.3f}')


axes[0].plot(range(start_plot, max_itt),bbvi.ELBO_history[start_plot:], label='ELBO')
axes[1].plot(range(start_plot, max_itt),bbvi.m_history[start_plot:, 0], '-', linewidth=0.5, label = "m")
axes[2].plot(range(start_plot, max_itt),bbvi.v_history[start_plot:, 0], '-', linewidth=0.5, label = "v")

number_of_param_to_plot = min(K,8)

for i in range(2,number_of_param_to_plot):
    axes[1].plot(range(start_plot, max_itt),bbvi.m_history[start_plot:, i], '-', linewidth=0.5)
    axes[2].plot(range(start_plot, max_itt),bbvi.v_history[start_plot:, i], '-', linewidth=0.5)

for i in range(3):
    axes.flat[i].legend(loc = 'lower right')

if save_fig:
    if random:
        plt.savefig(f'plots/Random_theta_K={K}_T={T}_batch_size={batch_size}_step_size={step_size}_acc={acc:.3}_Entropy={entropy:.3f}_ECE={ece:.3f}=LPD: {lpd:.3f}.png')
    else:
        plt.savefig(f'plots/Theta_MAP_K={K}_T={T}_batch_size={batch_size}_step_size={step_size}_acc={acc:.3}_Entropy={entropy:.3f}_ECE={ece:.3f}=LPD: {lpd:.3f}.png')

plt.show()