import numpy as np
import torch 
from models.LeNet5 import LeNet

softmax = lambda x: torch.exp(x - torch.max(x, dim=1, keepdim=True)[0]) / torch.sum(torch.exp(x - torch.max(x, dim=1, keepdim=True)[0]), dim=1, keepdim=True)


def compute_ECE(preds, ytest, num_bins = 10):
    
        # create bins
        bins = np.linspace(0, 1, num_bins+1)
        
        conf_all, yhat = torch.max(preds, dim=1)
        correct_all = 1.0*(ytest == yhat)
        
        # preallocate lists
        acc_bin, conf_bin, point_in_bins = [], [], []
        
        # loop through each bin
        for i in range(num_bins):
            bin_start, bin_end = bins[i], bins[i+1]        
            bin_idx = torch.logical_and(bin_start <= conf_all, conf_all < bin_end)
            num_points_in_bin = torch.sum(bin_idx)

            # don't want to bother with empty bins
            if num_points_in_bin == 0:
                continue
            
            # store results
            conf_bin.append(torch.mean(conf_all[bin_idx]).cpu())
            acc = torch.mean(correct_all[bin_idx])
            acc_bin.append(acc.cpu())
            point_in_bins.append(num_points_in_bin.cpu())

        acc_bin = np.array(acc_bin)
        conf_bin = np.array(conf_bin)
        point_in_bins = np.array(point_in_bins)

        # compute ECE
        ECE = np.sum(point_in_bins*np.abs(acc_bin-conf_bin))/len(ytest)
        
        return ECE
    
def compute_maximum_CE(preds, ytest, num_bins = 10):
    
    # create bins
    bins = np.linspace(0, 1, num_bins+1)
    
    conf_all, yhat = torch.max(preds, dim=1)
    correct_all = 1.0*(ytest == yhat)
    
    # preallocate lists
    acc_bin, conf_bin, point_in_bins = [], [], []
    
    # loop through each bin
    for i in range(num_bins):
        bin_start, bin_end = bins[i], bins[i+1]        
        bin_idx = torch.logical_and(bin_start <= conf_all, conf_all < bin_end)
        num_points_in_bin = torch.sum(bin_idx)

        # don't want to bother with empty bins
        if num_points_in_bin == 0:
            continue
        
        # store results
        conf_bin.append(torch.mean(conf_all[bin_idx]).cpu())
        acc = torch.mean(correct_all[bin_idx])
        acc_bin.append(acc.cpu())
        point_in_bins.append(num_points_in_bin.cpu())

    acc_bin = np.array(acc_bin)
    conf_bin = np.array(conf_bin)

    return np.max(np.abs(acc_bin-conf_bin))

def marginal_calibration_error(preds, ytest, num_bins = 10):
      # Calculate confidence and predicted class
    confidence, predictions = torch.max(preds, dim=1)
    
    # Initialize bins
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    # Initialize variables to store accuracy and confidence for each bin and class
    bin_acc = torch.zeros((num_bins, preds.size(1)))
    bin_conf = torch.zeros((num_bins, preds.size(1)))
    bin_count = torch.zeros((num_bins, preds.size(1)))
    
    # Calculate accuracy and confidence for each bin and class
    for i in range(num_bins):
        in_bin = (confidence > bin_lowers[i]) & (confidence <= bin_uppers[i])
        for c in range(preds.size(1)):
            class_in_bin = in_bin & (predictions == c)
            bin_count[i, c] = class_in_bin.sum().item()
            if bin_count[i, c] > 0:
                bin_acc[i, c] = (ytest[class_in_bin] == c).sum().item() / bin_count[i, c]
                bin_conf[i, c] = confidence[class_in_bin].mean().item()
                
    # Calculate marginal calibration error
    marginal_EC = torch.sum(bin_count*(bin_acc-bin_conf)**2).item()/len(ytest)
    
    return marginal_EC

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


print("Model predictions on MNIST") 
net = LeNet().to(device=device) 
weights = torch.load('./checkpoints/LeNet5_Mnist_acc_98.94_.pth') # load the weights
net.load_state_dict(weights)

net.eval()

predictions = softmax(net(xtest))

with torch.no_grad():

    accuracy = torch.sum(torch.argmax(predictions, dim=1) == ytest).float()/len(ytest)
    print(f"Accuracy: {accuracy:.3f}")


    entropy_posterior_predictive = -torch.sum(predictions * torch.log(predictions+1e-6), dim=1).mean().item()
    print(f"Entropy of predictions: {entropy_posterior_predictive:.3f}")


    lpd = torch.log(predictions[torch.arange(len(ytest)), ytest] + 1e-6).mean().item()
    print(f"Log Predictive Density: {lpd:.3f}")

    ece = compute_ECE(predictions, ytest)
    print(f"ECE: {ece:.3f}")
    
    max_CE = compute_maximum_CE(predictions, ytest)
    print(f"Maximum Calibration Error: {max_CE:.3f}")
    
    marginal_CE = marginal_calibration_error(predictions, ytest)
    print(f"Marginal Calibration Error: {marginal_CE:.3f}")
    
    

print("\n\n")
print("Model predictions on Fashion MNIST") 

xtrain, ytrain = torch.load('./datasets/fashion_mnist_train.pt')
xtest, ytest = torch.load('./datasets/fashion_mnist_test.pt')

xtrain = xtrain.to(device=device)
ytrain = ytrain.to(device=device)
xtest = xtest.to(device=device) 
ytest = ytest.to(device=device)


net = LeNet().to(device=device) 
weights = torch.load('./checkpoints/LeNet5_FashionMnist_acc_90.55_.pth') # load the weights
net.load_state_dict(weights)

net.eval()

predictions = softmax(net(xtest))

with torch.no_grad():

    accuracy = torch.sum(torch.argmax(predictions, dim=1) == ytest).float()/len(ytest)
    print(f"Accuracy: {accuracy:.3f}")


    entropy_posterior_predictive = -torch.sum(predictions * torch.log(predictions+1e-6), dim=1).mean().item()
    print(f"Entropy of predictions: {entropy_posterior_predictive:.3f}")


    lpd = torch.log(predictions[torch.arange(len(ytest)), ytest] + 1e-6).mean().item()
    print(f"Log Predictive Density: {lpd:.3f}")

    ece = compute_ECE(predictions, ytest)
    print(f"ECE: {ece:.3f}")
    
    max_CE = compute_maximum_CE(predictions, ytest)
    print(f"Maximum Calibration Error: {max_CE:.3f}")
    
    marginal_CE = marginal_calibration_error(predictions, ytest)
    print(f"Marginal Calibration Error: {marginal_CE:.3f}")
    
    
    
    


