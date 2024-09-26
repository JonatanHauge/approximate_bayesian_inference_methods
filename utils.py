import torch
import torch.optim as optim
from time import time
import numpy as np

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
        self.v = torch.zeros(num_params, requires_grad = True, device = self.device) #initialize v in log domain 
        self.optimizer = optim.Adam(params=[self.m, self.v], lr=step_size)
        
    def compute_ELBO(self):
         
        # generate samples from epsilon ~ N(0, 1) and use re-parametrization trick
        epsilon = torch.randn(self.num_params, device = self.device)
        z_samples = self.m  + torch.sqrt(torch.exp(self.v)) * epsilon  # shape:  (,K)
        w_samples = z_samples @ self.P.T + self.theta_map   # shape: (, D)
        expected_log_prior_term = torch.sum(self.log_prior(z_samples))  # shape: scalar
    
        # batch mode or minibatching?
        if self.batch_size:
            # Use mini-batching
            batch_idx = torch.randperm(self.N, device=self.device)[:self.batch_size]
            # Use the indices to create batches
            X_batch = self.X[batch_idx]
            y_batch = self.y[batch_idx]
              
            expected_log_lik_term = self.N/self.batch_size*self.log_lik(self.model, self.params, X_batch, y_batch, w_samples)  # shape: scalar
        else:
            # No mini-batching
            expected_log_lik_term = self.log_lik(self.X, self.y, w_samples)   # shape: scalar
        
        # compute ELBO
        ELBO = 1/self.T * expected_log_lik_term + expected_log_prior_term + self.compute_entropy(torch.exp(self.v))
        
        return -ELBO, [1/self.T * expected_log_lik_term, expected_log_prior_term, self.compute_entropy(torch.exp(self.v))]
    
    def generate_posterior_sample(self):
        with torch.no_grad():
            epsilon = torch.randn(self.num_params, device = self.device)
            z_sample = self.m + torch.sqrt(torch.exp(self.v)) * epsilon  # shape:  (,K)
            w_sample = z_sample @ self.P.T + self.theta_map
        return w_sample
    
    def predict(self, Xtest, num_samples=100):
        self.model.eval()
        y_preds = torch.zeros(len(Xtest), 10, device=self.device)
        with torch.no_grad():
            for i in range(num_samples):
                w_sample = self.generate_posterior_sample()
                set_weights(self.params, w_sample)
                y_preds += softmax(self.model(Xtest))
            y_preds /= num_samples
        return y_preds
    
    def compute_accuracy(self, Xtest, ytest, num_samples=100):
        y_preds = torch.argmax(self.predict(Xtest, num_samples), dim = 1)
        acc = torch.sum(y_preds == ytest).float()/len(ytest)
        return acc.cpu().detach().numpy()
    
    def compute_entropy_posterior_predictive(self, Xtest, num_samples=100):
        predictive_probs = self.predict(Xtest, num_samples)
        entropy = -torch.sum(predictive_probs * torch.log(predictive_probs+1e-6), dim=1).mean().cpu().item()
        return entropy
    
    
    def compute_ECE(self, Xtest, ytest, num_bins = 10):
        # inspiration from Bayesian Machine Learning course week 7
        preds = self.predict(Xtest)
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
    
    def compute_LPD(self, Xtest, ytest):
        preds = self.predict(Xtest)[torch.arange(len(ytest)), ytest]
        lpd = torch.log(preds + 1e-6).mean().cpu().item()
        return lpd
    
    def compute_entropy(self, v=None):
        """ Compute entropy term """
        entropy = 0.5 * torch.log(2 * torch.pi * v) + 0.5
        return entropy.sum()  

    def fit(self, X, y, seed=0):
        """ fits the variational approximation q given data (X,y) by maximizing the ELBO using gradient-based methods """ 
        torch.manual_seed(seed)
        self.X, self.y, self.N = X, y, len(X)
        self.ELBO_history, self.m_history, self.v_history = [], [], []
        self.log_like_history, self.log_prior_history, self.entropy_history = [], [], []
        
        print('Fitting approximation using BBVI')        
        t0 = time()
        for itt in range(self.max_itt):
            
            # evaluate ELBO
            ELBO, [log_like, prior, entropy] = self.compute_ELBO()
            
            # store current values for plotting purposes
            self.ELBO_history.append(-ELBO.clone().detach())
            self.m_history.append(self.m.clone().detach())
            self.v_history.append(torch.exp(self.v.clone().detach()))
            self.log_like_history.append(log_like.clone().detach())
            self.log_prior_history.append(prior.clone().detach())
            self.entropy_history.append(entropy.clone().detach())

            self.optimizer.zero_grad()
            ELBO.backward()
            self.optimizer.step() #SHould update self.lam
            
            # verbose?
            if self.verbose:
                if (itt+1) % 1000 == 0:
                    print('\tItt: %5d'% (itt)) 
        
        t1 = time()
        print('\tOptimization done in %3.2fs\n' % (t1-t0))
        
        # track quantities through iterations for visualization purposes
        self.ELBO_history = np.array([elbo.cpu().numpy() for elbo in self.ELBO_history]) #since we optimze the negative ELBO
        self.m_history = np.array([m.cpu().numpy() for m in self.m_history])
        self.v_history = np.array([v.cpu().numpy() for v in self.v_history])
        self.log_like_history = np.array([log_like.cpu().numpy() for log_like in self.log_like_history])
        self.log_prior_history = np.array([log_prior.cpu().numpy() for log_prior in self.log_prior_history])
        self.entropy_history = np.array([ent.cpu().numpy() for ent in self.entropy_history])
            
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