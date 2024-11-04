import numpy as np
import torch 
import sys
from models.LeNet5 import LeNet
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd.functional import hessian
from torchmetrics.classification import MulticlassCalibrationError
from torch.distributions.multivariate_normal import MultivariateNormal

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# helper functions 

def negative_log_like_NN_classification(last_params):
    logits = net.last_layer(z, last_params)
    nll = torch.nn.CrossEntropyLoss(reduction='sum')(logits, ytrain)
    return nll

def generate_posterior_weights(w_map, A):
    m = MultivariateNormal(w_map, A)
    weights = m.sample().to(device)
    return weights

def posterior_predict(model, w_map, A, xtest, num_samples = 100, num_classes = 10):
    model.eval()
    N = len(xtest)
    y_preds = torch.zeros(N, num_classes, device=device) #hardcoded number of labels (100)
    with torch.no_grad():
        for _ in range(num_samples):
            w_LL = generate_posterior_weights(w_map, A)
            z = model.first_layers(xtest)
            y_preds += torch.nn.functional.softmax(model.last_layer(z, w_LL), dim = 1)
    return y_preds / num_samples

def compute_all_metrics(model, w_map, A, xtest, ytest, x_ood, num_samples=100, num_bins=10):
    """ Compute all metrics """
    logits = posterior_predict(model, w_map, A, xtest, num_samples=num_samples, num_classes=10)
    logits_ood = posterior_predict(model, w_map, A, x_ood, num_samples=num_samples, num_classes=10)
    acc = torch.sum(torch.argmax(logits, dim=1) == ytest).float().mean().cpu().item() / len(ytest)
    entropy = -torch.sum(logits * torch.log(logits+1e-6), dim=1).mean().cpu().item()
    ood_entropy = -torch.sum(logits_ood * torch.log(logits_ood+1e-6), dim=1).mean().cpu().item()
    lpd = torch.log(logits[torch.arange(len(ytest)), ytest] + 1e-6).mean().cpu().item()
    ece = MulticlassCalibrationError(num_classes=10, n_bins=num_bins, norm='l1')(logits, ytest).cpu().item()
    mce = MulticlassCalibrationError(num_classes=10, n_bins=num_bins, norm='max')(logits, ytest).cpu().item()
    return acc, entropy, lpd, ece, mce, ood_entropy
    


# settings
sys.stdout.flush()
datasets = ['mnist','fashion_mnist']
prior_sigma_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
for dataset in datasets:
    result_table = pd.DataFrame(index = ['Accuracy', 'Entropy', 'LPD', 'ECE', 'MCE', 'OOD'])
    # Load data
    if dataset == 'mnist':
        xtrain, ytrain = torch.load('./datasets/mnist_train.pt', map_location=device)
        xtest, ytest = torch.load('./datasets/mnist_test.pt', map_location=device)
        weights = torch.load('./checkpoints/LeNet5_Mnist_acc_98.94_.pth', map_location=device) # load the weights
        x_ood, y_ood = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
    elif dataset == 'fashion_mnist':
        xtrain, ytrain = torch.load('./datasets/fashion_mnist_train.pt', map_location=device)
        xtest, ytest = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
        weights = torch.load('./checkpoints/LeNet5_FashionMnist_acc_90.55_.pth', map_location=device)
        x_ood, y_ood = torch.load('./datasets/mnist_test.pt', map_location=device)
    
    net = LeNet().to(device=device) 
    net.load_state_dict(weights)
    z = net.first_layers(xtrain)

    
    last_params_map = torch.cat([weights['fc3.weight'].flatten(), weights['fc3.bias'].flatten()])

    H_neg_log_like = hessian(negative_log_like_NN_classification, last_params_map, create_graph=True)
    print("Hessian computed")
    
    for prior_sigma in prior_sigma_values:       
        H_prior = torch.eye(last_params_map.shape[0], device=device) * 1/prior_sigma
        H = H_neg_log_like + H_prior
        A = torch.linalg.inv(H)
        print("Hessian inverted")
        is_symmetric = torch.allclose(A, A.T)
        num_samples = 100
        if is_symmetric:
            acc, entropy, lpd, ece, mce, ood = compute_all_metrics(net, last_params_map, A, xtest, ytest, x_ood, num_samples=100, num_bins=10)

        else:
            # when prior_sigma is large >= 1e-1, there is numerical instability in the construction of A 
            # (from the inverse operation) which has the effect that it is not symmetric
            
            A = (A+A.T)/2 # make sure it is symmetric
            acc, entropy, lpd, ece, mce, ood = compute_all_metrics(net, last_params_map, A, xtest, ytest, x_ood, num_samples=100, num_bins=10)

            
        result_table[f"prior_sigma={prior_sigma}"] = [acc, entropy, lpd, ece, mce, ood]    
    
    result_table.to_csv(f"LLLA/{dataset}.csv")
