from utils import BlackBoxVariationalInference, log_like_NN_classification
import numpy as np
import torch 
import sys
from models.LeNet5 import LeNet
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd.functional import hessian
from torchmetrics.classification import MulticlassCalibrationError


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


# settings
sys.stdout.flush()
print('BBVI job is now running')
datasets = ['fashion_mnist', 'mnist']
for dataset in datasets:
    result_table = pd.DataFrame(index = ['Accuracy', 'Entropy', 'LPD', 'ECE', 'MCE', 'OOD'])
    prior_sigma = 0.001
    verbose = True
    save_fig = True
    batch_size = 500

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
        weights = torch.load('./checkpoints/LeNet5_Mnist_acc_98.94%.pth') # load the weights
    elif dataset == 'fashion_mnist':
        xtrain, ytrain = torch.load('./datasets/fashion_mnist_train.pt', map_location=device)
        train_loader = DataLoader(TensorDataset(xtrain, ytrain), batch_size=batch_size, shuffle=True)
        xtest, ytest = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
        test_loader = DataLoader(TensorDataset(xtest, ytest), batch_size=100, shuffle=False)
        weights = torch.load('./checkpoints/LeNet5_FashionMnist_acc_90.55%.pth')

    assert len(test_loader.dataset) % test_loader.batch_size == 0, 'Test loader batch size must divide test set size'

    net = LeNet().to(device=device) 
    num_params = sum(p.numel() for p in net.parameters())
    print('Number of parameters:', num_params)
    params = extract_parameters(net)

    def negative_log_like_NN_classification(weights):
        set_weights(params, weights)
        nll = 0
        for X, y in train_loader:
            nll += torch.nn.CrossEntropyLoss(reduction='sum')(net(X), y)
        return nll
    
    def generate_posterior_weights(w_map, A):
        weights = np.random.multivariate_normal(w_map, A, 1)
        return torch.Tensor(weights).to(device)
    
    def posterior_predict(model, w_map, A, test_loader, num_samples = 100, num_classes = 10):
        model.eval()
        N = len(test_loader.dataset)
        y_preds = torch.zeros(N, num_classes, device=device) #hardcoded number of labels (100)
        with torch.no_grad():
            for _ in range(num_samples):
                weights = generate_posterior_weights(w_map, A)
                set_weights(params, weights)
                idx = 0
                for X, y in test_loader:
                    X = X.to(device)
                    batch_size = len(X)
                    y_preds[idx:idx+batch_size, :] += torch.nn.functional.softmax(model(X))
                    idx += batch_size
            y_preds /= num_samples
        return y_preds
    
    def compute_all_metrics(model, w_map, A, test_loader, num_samples=100, num_bins=10):
        """ Compute all metrics """
        logits = posterior_predict(model, w_map, A, test_loader, num_samples=num_samples, num_classes=10)
        ytest = torch.cat([y for _, y in test_loader], dim=0)
        ytest = ytest.to(device)
        acc = torch.sum(torch.argmax(logits, dim=1) == ytest).float().mean().cpu().item() / len(ytest)
        entropy = -torch.sum(logits * torch.log(logits+1e-6), dim=1).mean().cpu().item()
        lpd = torch.log(logits[torch.arange(len(ytest)), ytest] + 1e-6).mean().cpu().item()
        ece = MulticlassCalibrationError(num_classes=10, n_bins=num_bins, norm='l1')(logits, ytest).cpu().item()
        mce = MulticlassCalibrationError(num_classes=10, n_bins=num_bins, norm='max')(logits, ytest).cpu().item()
        return acc, entropy, lpd, ece, mce
    

    weights = torch.cat([w.flatten() for w in weights.values()]) # flatten the weights 

    H_neg_log_like = hessian(negative_log_like_NN_classification, weights)
    H_prior = torch.eye(weights.shape[0]) * 1/prior_sigma
    H = H_neg_log_like + H_prior
    A = np.linalg.inv(H.detach().numpy())

    print(H_neg_log_like.shape)
    print('Hessian computed')

    num_samples = 100
    acc, entropy, lpd, ece, mce = compute_all_metrics(net, weights, A, test_loader, num_samples=100, num_bins=10)

    print('Accuracy:', np.round(acc, 3))
    print('Entropy:', np.round(entropy, 3))
    print('LPD:', np.round(lpd, 3))
    print('ECE:', np.round(ece, 3))
    print('MCE:', np.round(mce, 3))

    result_table[f"prior_sigma={prior_sigma}"] = [acc, entropy, lpd, ece, mce]
    print(result_table)