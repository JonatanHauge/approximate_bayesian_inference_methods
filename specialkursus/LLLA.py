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

# settings
sys.stdout.flush()
datasets = ['fashion_mnist', 'mnist']
for dataset in datasets:
    result_table = pd.DataFrame(index = ['Accuracy', 'Entropy', 'LPD', 'ECE', 'MCE', "OOD"])
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
        xtest, ytest = torch.load('./datasets/mnist_test.pt', map_location=device)
        weights = torch.load('./checkpoints/LeNet5_Mnist_acc_98.94%.pth') # load the weights
        x_ood, y_ood = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
    elif dataset == 'fashion_mnist':
        xtrain, ytrain = torch.load('./datasets/fashion_mnist_train.pt', map_location=device)
        xtest, ytest = torch.load('./datasets/fashion_mnist_test.pt', map_location=device)
        weights = torch.load('./checkpoints/LeNet5_FashionMnist_acc_90.55%.pth')
        x_ood, y_ood = torch.load('./datasets/mnist_test.pt', map_location=device)

    net = LeNet().to(device=device) 
    net.load_state_dict(weights)
    z = net.first_layers(xtrain)

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
    
    
    last_params_map = torch.cat([weights['fc3.weight'].flatten(), weights['fc3.bias'].flatten()])

    H_neg_log_like = hessian(negative_log_like_NN_classification, last_params_map, create_graph=True)
    H_prior = torch.eye(last_params_map.shape[0]) * 1/prior_sigma
    H = H_neg_log_like + H_prior
    print("Hessian computed")
    A = torch.linalg.inv(H)
    print("Hessian inverted")

    num_samples = 100
    acc, entropy, lpd, ece, mce, ood = compute_all_metrics(net, last_params_map, A, xtest, ytest, x_ood, num_samples=100, num_bins=10)

    print('Accuracy:', np.round(acc, 3))
    print('Entropy:', np.round(entropy, 3))
    print('LPD:', np.round(lpd, 3))
    print('ECE:', np.round(ece, 3))
    print('MCE:', np.round(mce, 3))
    print('OOD:', np.round(ood, 3))

    result_table[f"{dataset}, prior_sigma={prior_sigma}"] = [acc, entropy, lpd, ece, mce, ood]
    print(result_table)
    print("done")