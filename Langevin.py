import sys

# libraries
import numpy as np
import torch 
import pandas as pd
from models.LeNet5 import LeNet
from torchmetrics.classification import MulticlassCalibrationError
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# Helper functions 

log_npdf = lambda x, m, v: -(x - m) ** 2 / (2 * v) - 0.5 * torch.log(2 * torch.pi * v)
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

def log_prior_function(z, prior_sigma = 0.1):
    log_prior = torch.sum(log_npdf(z, torch.tensor(0), torch.tensor(prior_sigma))) #Assume prior is N(0, prior_sigma)
    return log_prior
    

def log_like_NN_classification(model, params, X, y, theta):

    set_weights(params, theta)# Set the weights for the model
    nll = torch.nn.CrossEntropyLoss(reduction='sum')(model(X), y)

    return -nll


# user input 
K = int(sys.argv[1])
T = int(sys.argv[2])
prior_sigma = float(sys.argv[3])


# setup parameters
torch.manual_seed(4242)
batch_size = 1000
epochs = 1000
num_posterior_samples = 1000
save_fig = True
random = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


xtrain, ytrain = torch.load('./datasets/mnist_train.pt', map_location=device)
xtest, ytest = torch.load('./datasets/mnist_test.pt', map_location=device)
weights = torch.load('./checkpoints/LeNet5_Mnist_acc_98.94_.pth') # load the weights


train_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, shuffle=False)


# set up P and theta

net = LeNet().to(device=device) 
theta_map = torch.cat([w.flatten() for w in weights.values()]) 
if random:
    theta_map = torch.zeros_like(theta_map)
theta_map = theta_map.to(device=device)

num_params = sum(p.numel() for p in net.parameters())
P = torch.randn(num_params, K, device=device)
P /= torch.norm(P, dim=0)  # Normalize columns to norm 1



params = extract_parameters(net)
z = torch.randn(K, device=device, requires_grad=True)

N = len(train_loader.dataset)

# function to schedule the learning rate
def compute_lr(t, start=0.0001, end=0.000001):
    gamma = 0.55
    b = (epochs * N / batch_size) / ((end / start) ** (-1/gamma) - 1) 
    a = start * (b ** gamma)
    return a * ((b + t) ** (-gamma))

def sample_sgld(batch_size, z, net, P, log_prior_function, log_like_NN_classification):
    
    lr = compute_lr(0)
    samples = []
    for epoch in range(epochs):
        for i, batch_data in enumerate(train_loader, 0):
           # z.grad.zero_()
            X, y = batch_data
            X, y = X.to(device), y.to(device)
            
            
            theta = z @ P.T + theta_map
            log_like = log_like_NN_classification(net, params, X, y, theta)
            log_prior = log_prior_function(z, prior_sigma)
            loss = -(N/batch_size*log_like*(1/T) + log_prior)
            loss.backward()

            epsilon = torch.randn(K, device = device)                       
            
            lr = compute_lr((N/batch_size) * epoch + i) #, start=lr_start, end=lr_end)
            
          
            z = z - lr * z.grad + torch.sqrt(2*torch.tensor(lr)) * epsilon
            samples.append(z.detach().clone())
            
            z = z.detach().clone().requires_grad_(True)
            
        if epoch % 50 == 0:
            print(f'[{epoch+1}] loss: {loss.item():.3f}, lr: {lr:.6f}')
    
    return samples, lr

samples, lr_end = sample_sgld(batch_size=batch_size, z=z, net=net, 
                      P=P, log_prior_function=log_prior_function, 
                      log_like_NN_classification=log_like_NN_classification)


posterior_samples = samples[-num_posterior_samples:]

# function to predict  
def predict(test_loader, samples, model, num_labels = 10):
    model.eval()
    N = len(test_loader.dataset)
    y_preds = torch.zeros(N, num_labels, device=device) 
    with torch.no_grad():
        for sample in range(len(samples)):
            z_sample = samples[sample]
            w_sample = z_sample @ P.T + theta_map
            set_weights(params, w_sample)
            idx = 0
            for Xtest, _ in test_loader:
                Xtest = Xtest.to(device)
                batch_size = len(Xtest)
                y_preds[idx:idx+batch_size, :] += softmax(model(Xtest))
                idx += batch_size
        y_preds /= len(samples)
    return y_preds

print("Make predictions")

predictions = predict(test_loader, posterior_samples, net, num_labels = 10)


def compute_all_metrics(test_loader, predictions, num_bins=10,num_labels=10):
    """ Compute all metrics """
    ytest = torch.cat([y for _, y in test_loader], dim=0)
    ytest = ytest.to(device)
    acc = torch.sum(torch.argmax(predictions, dim=1) == ytest).float().mean().cpu().item() / len(ytest)
    entropy = -torch.sum(predictions * torch.log(predictions+1e-6), dim=1).mean().cpu().item()
    lpd = torch.log(predictions[torch.arange(len(ytest)), ytest] + 1e-6).mean().cpu().item()
    ece = MulticlassCalibrationError(num_classes=num_labels, n_bins=num_bins, norm='l1')(predictions, ytest).cpu().item()
    mce = MulticlassCalibrationError(num_classes=num_labels, n_bins=num_bins, norm='max')(predictions, ytest).cpu().item()
    return acc, entropy, lpd, ece, mce


print("compute metrics")
acc, entropy, lpd, ece, mce = compute_all_metrics(test_loader, predictions, num_bins=10)

print('Accuracy:', np.round(acc, 3))
print('Entropy:', np.round(entropy, 3))
print('LPD:', np.round(lpd, 3))
print('ECE:', np.round(ece, 3))
print('MCE:', np.round(mce, 3))



xtest, ytest = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, shuffle=False)
predictions = predict(test_loader, posterior_samples, net, num_labels = 10)
ood = -torch.sum(predictions * torch.log(predictions+1e-6), dim=1).mean().cpu().item()


print("OOD:", np.round(ood,3))


metrics = {
    "Accuracy": [np.round(acc, 3)],
    "Entropy": [np.round(entropy, 3)],
    "LPD": [np.round(lpd, 3)],
    "ECE": [np.round(ece, 3)],
    "MCE": [np.round(mce, 3)],
    "OOD": [np.round(ood,3)],
    "LR start": [compute_lr(0)],
    "LR end": [lr_end]
}

df = pd.DataFrame(metrics)
df.to_csv(f"Langevin_output/MetricsMnist_K={K}_T={T}_prior_sigma={prior_sigma}.csv", index=False)


# PLOTTING
samples_plot = torch.stack(samples).cpu().numpy()

number_of_dim_to_plot = min(K,10)

# Plot the trajectory
plt.figure(figsize=(10, 6))

for i in range(number_of_dim_to_plot):
    plt.plot(samples_plot[:, i], linestyle = '--')
    
plt.title('z samples')

plt.grid(True)

if random:
    plt.savefig(f'Langevin_output/MNIST_random_K={K}_T={T}_prior_sigma={prior_sigma}.png')
else:
    plt.savefig(f'Langevin_output/MNIST_MAP_K={K}_T={T}_prior_sigma={prior_sigma}.png')


### Re-do on Fashion MNIST
K = int(sys.argv[4])
T = int(sys.argv[5])
prior_sigma = float(sys.argv[6])


xtrain, ytrain = torch.load('./datasets/fashion_mnist_train.pt', map_location=device)
xtest, ytest = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
weights = torch.load('./checkpoints/LeNet5_FashionMnist_acc_90.55_.pth')



train_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, shuffle=False)


# set up P and theta

net = LeNet().to(device=device) 
theta_map = torch.cat([w.flatten() for w in weights.values()]) 
if random:
    theta_map = torch.zeros_like(theta_map)
theta_map = theta_map.to(device=device)

num_params = sum(p.numel() for p in net.parameters())
P = torch.randn(num_params, K, device=device)
P /= torch.norm(P, dim=0)  # Normalize columns to norm 1



params = extract_parameters(net)
z = torch.randn(K, device=device, requires_grad=True)

N = len(train_loader.dataset)

samples, lr_end = sample_sgld(batch_size=batch_size, z=z, net=net, 
                      P=P, log_prior_function=log_prior_function, 
                      log_like_NN_classification=log_like_NN_classification)



posterior_samples = samples[-num_posterior_samples:]


print("Make predictions")

predictions = predict(test_loader, posterior_samples, net, num_labels = 10)



acc, entropy, lpd, ece, mce = compute_all_metrics(test_loader, predictions, num_bins=10)

print('Accuracy:', np.round(acc, 3))
print('Entropy:', np.round(entropy, 3))
print('LPD:', np.round(lpd, 3))
print('ECE:', np.round(ece, 3))
print('MCE:', np.round(mce, 3))



xtest, ytest = torch.load('./datasets/mnist_test.pt', map_location=device)
test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, shuffle=False)
predictions = predict(test_loader, posterior_samples, net, num_labels = 10)
ood = -torch.sum(predictions * torch.log(predictions+1e-6), dim=1).mean().cpu().item()


print("OOD:", np.round(ood,3))


metrics = {
    "Accuracy": [np.round(acc, 3)],
    "Entropy": [np.round(entropy, 3)],
    "LPD": [np.round(lpd, 3)],
    "ECE": [np.round(ece, 3)],
    "MCE": [np.round(mce, 3)],
    "OOD": [np.round(ood,3)],
    "LR start": [compute_lr(0)],
    "LR end": [lr_end]
}

df = pd.DataFrame(metrics)
df.to_csv(f"Langevin_output/MetricsFashionMnist_K={K}_T={T}_prior_sigma={prior_sigma}.csv", index=False)


# PLOTTING
samples_plot = torch.stack(samples).cpu().numpy()

number_of_dim_to_plot = min(K,10)

# Plot the trajectory
plt.figure(figsize=(10, 6))

for i in range(number_of_dim_to_plot):
    plt.plot(samples_plot[:, i], linestyle = '--')
    
plt.title('z samples')

plt.grid(True)

if random:
    plt.savefig(f'Langevin_output/FashionMNIST_random_K={K}_T={T}_prior_sigma={prior_sigma}.png')
else:
    plt.savefig(f'Langevin_output/FashionMNIST_MAP_K={K}_T={T}_prior_sigma={prior_sigma}.png')
