import torch
import torch.optim as optim
from time import time
import numpy as np
from torchmetrics.classification import MulticlassCalibrationError
from tqdm import tqdm

log_npdf = lambda x, m, v: -(x - m) ** 2 / (2 * v) - 0.5 * torch.log(2 * torch.pi * v)
softmax = lambda x: torch.exp(x - torch.max(x, dim=1, keepdim=True)[0]) / torch.sum(torch.exp(x - torch.max(x, dim=1, keepdim=True)[0]), dim=1, keepdim=True) 
    
class BlackBoxVariationalInference(object):
    def __init__(self, model, theta_map, P, log_lik, num_params, step_size=1e-2, max_itt=2000, batch_size=None, seed=0, verbose=False, T = 1000, prior_sigma = 1, device = None):
        
        self.model = model                      # model
        self.params = extract_parameters(model) # extract parameters
        self.log_lik = log_lik              # log likelihood function
        self.batch_size = batch_size        # batch size
        self.seed = seed                    # seed
        self.P = P                          # random projection matrix
        self.theta_map = theta_map          # MAP estimate
        self.T = T                          # Temperature parameter
        self.num_params = num_params        # number of parameters
        self.verbose = verbose        
        self.X, self.y, self.ELBO = None, None, None
        self.num_params, self.step_size, self.max_itt = num_params, step_size, max_itt
        self.num_var_params = 2*self.num_params  
        self.prior_sigma = torch.tensor(prior_sigma)
        self.device = device 
        
        # set   parameters and optimizer
        self.m = torch.zeros(num_params, requires_grad=True, device = self.device) 
        self.v = torch.tensor([-1. for _ in range(num_params)], requires_grad = True, device = self.device) #initialize v in log domain 
        self.optimizer = optim.Adam(params=[self.m, self.v], lr=step_size)
        
    def compute_ELBO(self, X, y):
         
        # generate samples from epsilon ~ N(0, 1) and use re-parametrization trick
        X, y = X.to(self.device), y.to(self.device)
        batch_size = len(X)
        epsilon = torch.randn(self.num_params, device = self.device)
        z_samples = self.m  + torch.sqrt(torch.exp(self.v)) * epsilon  # shape:  (,K)
        w_samples = z_samples @ self.P.T + self.theta_map   # shape: (, D)
        expected_log_prior_term = torch.sum(self.log_prior(z_samples))  # shape: scalar
              
        expected_log_lik_term = self.N/batch_size*self.log_lik(self.model, self.params, X, y, w_samples)  # shape: scalar
        # compute ELBO
        ELBO = 1/self.T * expected_log_lik_term + expected_log_prior_term + self.compute_entropy(torch.exp(self.v))
        
        return -ELBO, [1/self.T * expected_log_lik_term, expected_log_prior_term, self.compute_entropy(torch.exp(self.v))]
    
    def generate_posterior_sample(self):
        with torch.no_grad():
            epsilon = torch.randn(self.num_params, device = self.device)
            z_sample = self.m + torch.sqrt(torch.exp(self.v)) * epsilon  # shape:  (,K)
            w_sample = z_sample @ self.P.T + self.theta_map
        return w_sample
    
    def predict(self, test_loader, num_samples=100):
        self.model.eval()
        N = len(test_loader.dataset)
        y_preds = torch.zeros(N, 100, device=self.device) #hardcoded number of labels (100)
        with torch.no_grad():
            for _ in range(num_samples):
                w_sample = self.generate_posterior_sample()
                set_weights(self.params, w_sample)
                idx = 0
                for Xtest, _ in test_loader:
                    Xtest = Xtest.to(self.device)
                    batch_size = len(Xtest)
                    y_preds[idx:idx+batch_size, :] += softmax(self.model(Xtest))
                    idx += batch_size
            y_preds /= num_samples
        return y_preds
    
    def compute_accuracy(self, test_loader, num_samples=100):
        """ Compute accuracy """
        y_preds = torch.argmax(self.predict(test_loader, num_samples), dim = 1)
        ytest = torch.cat([y for _, y in test_loader], dim=0)
        ytest = ytest.to(self.device)
        acc = torch.sum(y_preds == ytest).float().mean().cpu().item() / len(ytest)
        return acc
    
    def compute_entropy_posterior_predictive(self, test_loader, num_samples=100):
        """ Compute entropy of the posterior predictive distribution """
        predictive_probs = self.predict(test_loader, num_samples)
        entropy = -torch.sum(predictive_probs * torch.log(predictive_probs+1e-6), dim=1).mean().cpu().item()
        return entropy
    
    def compute_ECE(self, test_loader, num_bins = 10):
        """ Compute expected calibration error """
        preds = self.predict(test_loader)
        metric = MulticlassCalibrationError(num_classes=100, n_bins=num_bins, norm='l1')
        ytest = torch.cat([y for _, y in test_loader], dim=0)
        ytest = ytest.to(self.device)
        ece = metric(preds, ytest).cpu().item()
        return ece
    
    def compute_MCE(self, test_loader, num_bins = 10):
        """ Compute maximum calibration error """
        preds = self.predict(test_loader)
        metric = MulticlassCalibrationError(num_classes=100, n_bins=num_bins, norm='max')
        ytest = torch.cat([y for _, y in test_loader], dim=0)
        ytest = ytest.to(self.device)
        ece = metric(preds, ytest).cpu().item()
        return ece
    
    def compute_LPD(self, test_loader):
        """ Compute log predictive density """
        ytest = torch.cat([y for _, y in test_loader], dim=0)
        ytest = ytest.to(self.device)
        preds = self.predict(test_loader)[torch.arange(len(ytest)), ytest]
        lpd = torch.log(preds + 1e-6).mean().cpu().item()
        return lpd
    
    def compute_all_metrics(self, test_loader, num_samples=100, num_bins=10):
        """ Compute all metrics """
        logits = self.predict(test_loader, num_samples=num_samples)
        ytest = torch.cat([y for _, y in test_loader], dim=0)
        ytest = ytest.to(self.device)
        acc = torch.sum(torch.argmax(logits, dim=1) == ytest).float().mean().cpu().item() / len(ytest)
        entropy = -torch.sum(logits * torch.log(logits+1e-6), dim=1).mean().cpu().item()
        lpd = torch.log(logits[torch.arange(len(ytest)), ytest] + 1e-6).mean().cpu().item()
        ece = MulticlassCalibrationError(num_classes=100, n_bins=num_bins, norm='l1')(logits, ytest).cpu().item()
        mce = MulticlassCalibrationError(num_classes=100, n_bins=num_bins, norm='max')(logits, ytest).cpu().item()
        return acc, entropy, lpd, ece, mce
    
    def compute_entropy(self, v=None):
        """ Compute entropy term """
        entropy = 0.5 * torch.log(2 * torch.pi * v) + 0.5
        return entropy.sum()  

    def fit(self, train_loader, seed=0):
        """ fits the variational approximation q given data (X,y) by maximizing the ELBO using gradient-based methods """ 
        torch.manual_seed(seed)
        self.N = len(train_loader.dataset)
        self.ELBO_history, self.m_history, self.v_history = [], [], []
        self.log_like_history, self.log_prior_history, self.entropy_history = [], [], []
        
        print('Fitting approximation using BBVI')        
        t0 = time()
        for itt in tqdm(range(self.max_itt), desc='Training Progress', leave=True):
            X, y = next(iter(train_loader))
            X, y = X.to(self.device), y.to(self.device)
            
            ELBO, [log_like, prior, entropy] = self.compute_ELBO(X, y) # evaluate ELBO
            
            # store current values for plotting purposes
            self.ELBO_history.append(-ELBO.clone().detach().cpu().numpy())
            self.m_history.append(self.m.clone().detach().cpu().numpy())
            self.v_history.append(torch.exp(self.v.clone().detach()).cpu().numpy())
            self.log_like_history.append(log_like.clone().detach().cpu().numpy())
            self.log_prior_history.append(prior.clone().detach().cpu().numpy())
            self.entropy_history.append(entropy.clone().detach().cpu().numpy())

            self.optimizer.zero_grad()
            ELBO.backward()
            self.optimizer.step() #SHould update self.lam
            
            # verbose?
            if self.verbose and (itt + 1) % 100 == 0:  # Update every 100 iterations
                tqdm.write(f'\tIteration: {itt + 1}, ELBO: {np.mean(self.ELBO_history[itt-100:itt]):.2f}, Log Likelihood: {np.mean(self.log_like_history[itt-100:itt]):.2f}, '
                       f'Prior: {np.mean(self.log_prior_history[itt-100:itt]):.2f}, Entropy: {np.mean(self.entropy_history[itt-100:itt]):.2f}')
        
        t1 = time()
        print('\tOptimization done in %3.2fs\n' % (t1-t0))
        
        # track quantities through iterations for visualization purposes
        self.ELBO_history = np.array(self.ELBO_history) #since we optimze the negative ELBO
        self.m_history = np.array(self.m_history)
        self.v_history = np.array(self.v_history)
        self.log_like_history = np.array(self.log_like_history)
        self.log_prior_history = np.array(self.log_prior_history)
        self.entropy_history = np.array(self.entropy_history)
            
        return self
    
    def log_prior(self, z):
        log_prior = torch.sum(log_npdf(z, torch.tensor(0), self.prior_sigma)) #Assume prior is N(0, prior_sigma)
        return log_prior


def log_like_NN_classification(model, params, X, y, theta):

    set_weights(params, theta)# Set the weights for the model
    nll = torch.nn.CrossEntropyLoss(reduction='sum')(model(X), y)

    return -nll

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