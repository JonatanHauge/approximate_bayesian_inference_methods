import autograd.numpy as np
from torch.autograd import grad
import torch
import torch.optim as optim
from time import time

log_npdf = lambda x, m, v: -(x - m) ** 2 / (2 * v) - 0.5 * torch.log(2 * torch.pi * v)
log_mvnpdf = lambda x, m, v: -0.5 * torch.sum((x - m) ** 2 / v + torch.log(2 * torch.pi * v), dim=1)
softmax = lambda x: torch.exp(x - torch.max(x, dim=1, keepdim=True)[0]) / torch.sum(torch.exp(x - torch.max(x, dim=1, keepdim=True)[0]), dim=1, keepdim=True)

class VariationalInference(object):
    
    def __init__(self, num_params, step_size=1e-2, max_itt=2000, verbose=False, name='VariationalInference'):
        
        self.name = name
        self.verbose = verbose        
        self.X, self.y, self.ELBO = None, None, None
        self.num_params, self.step_size, self.max_itt = num_params, step_size, max_itt
        self.num_var_params = 2*self.num_params
        
        # Initialize the variational parameters for the mean-field approximation
        self.m, self.v = torch.zeros(num_params, requires_grad=True), torch.zeros(num_params, requires_grad=True) #initialize v in log domain   
        # prepare optimizer
        self.optimizer = optim.Adam(params=[self.m, self.v], lr=step_size)
        
    def compute_entropy(self, v=None):
        """ Compute entropy term """
        entropy = 0.5 * torch.log(2 * torch.pi * v) + 0.5
        return entropy.sum()  

    def generate_posterior_samples(self, num_samples=1000):
        mean = self.m
        var = self.v
        std = torch.sqrt(var)
        samples = torch.normal(mean=mean.unsqueeze(0), std=std.unsqueeze(0).expand(num_samples, -1))
        return samples

    def fit(self, X, y, seed=0):
        """ fits the variational approximation q given data (X,y) by maximizing the ELBO using gradient-based methods """ 
        torch.manual_seed(seed)
        self.X, self.y, self.N = X, y, len(X)
        self.ELBO_history, self.m_history, self.v_history = [], [], []
        self.log_like_history, self.log_prior_history, self.entropy_history = [], [], []
        
        print('Fitting approximation using %s' % self.name)        
        t0 = time()
        for itt in range(self.max_itt):
            
            # evaluate ELBO
            ELBO, [log_like, prior, entropy] = self.compute_ELBO()
            
            # store current values for plotting purposes
            self.ELBO_history.append(-ELBO.clone().detach().numpy())
            self.m_history.append(self.m.clone().detach().numpy())
            self.v_history.append(torch.exp(self.v.clone().detach()).numpy())
            self.log_like_history.append(log_like.clone().detach().numpy())
            self.log_prior_history.append(prior.clone().detach().numpy())
            self.entropy_history.append(entropy.clone().detach().numpy())

            self.optimizer.zero_grad()
            ELBO.backward()
            self.optimizer.step() #SHould update self.lam
            
            # verbose?
            if self.verbose:
                if (itt+1) % 250 == 0:
                    print('\tItt: %5d, ELBO = %3.2f' % (itt, np.mean(self.ELBO_history[-250:])))
        
        t1 = time()
        print('\tOptimization done in %3.2fs\n' % (t1-t0))
        
        # track quantities through iterations for visualization purposes
        self.ELBO_history = torch.tensor(self.ELBO_history) #since we optimze the negative ELBO
        self.m_history = np.array(self.m_history)
        self.v_history = np.array(self.v_history)
        self.log_like_history = np.array(self.log_like_history)
        self.log_prior_history = np.array(self.log_prior_history)
        self.entropy_history = np.array(self.entropy_history)
            
        return self
    
class BlackBoxVariationalInference(VariationalInference):
    def __init__(self, model, theta_map, P, log_prior, log_lik, num_params, step_size=1e-2, max_itt=2000, batch_size=None, seed=0, verbose=False, T = 1000):
    
        # arguments specific to BBVI
        self.model = model                  # model
        self.params = extract_parameters(model) # extract parameters
        self.log_prior = log_prior          # function for evaluating the log prior
        self.log_lik = log_lik              # function for evaluating the log likelihood
        self.batch_size = batch_size        # batch size
        self.seed = seed                    # seed
        self.P = P                          # random projection matrix
        self.theta_map = theta_map          # MAP estimate
        self.T = T                          # Temperature parameter
        self.num_params = num_params        # number of parameters
        
        # pass remaining argument to VI class
        super(BlackBoxVariationalInference, self).__init__(name="Black-box VI", num_params=num_params, step_size=step_size, max_itt=max_itt, verbose=verbose)
    
    def compute_ELBO(self):
        """ compute an estimate of the ELBO for variational parameters in the *lam*-variable using Monte Carlo samples and the reparametrization trick.
            
            If self.batch_size is None, the function computes the ELBO using full dataset. If self.batch is not None, the function estimates the ELBO using a single minibatch of size self.batch_size. 
            The samples in the mini batch is sampled uniformly from the full dataset.
            
            inputs:
            lam       --     vector of variational parameters (unconstrained space)

            outputs:
            ELBO      --     estimate of the ELBO (scalar)

            """
        
        # generate samples from epsilon ~ N(0, 1) and use re-parametrization trick
        epsilon = torch.randn(self.num_params)
        z_samples = self.m + torch.sqrt(torch.exp(self.v)) * epsilon  # shape:  (,K)
        w_samples = z_samples @ self.P.T + self.theta_map   # shape: (, D)
        expected_log_prior_term = torch.mean(self.log_prior(z_samples))  # shape: scalar
    
        # batch mode or minibatching?
        if self.batch_size:
            # Use mini-batching
            batch_idx = torch.randperm(self.N)[:self.batch_size]
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



def log_like_NN_classification(model, params, X, y, theta):
    """
    Implements the log likelihood function for the classification NN with categorical likelihood.
    S is number of MC samples, N is number of datapoints in likelihood and D is the dimensionality of the model (number of weights).

    Inputs:
    X              -- Data (np.array of size N x D)
    y              -- vector of target (np.array of size N)
    theta_s        -- vector of weights (np.array of size (S, D))

    outputs: 
    log_likelihood -- Array of log likelihood for each sample in z (np.array of size S)
     """

    set_weights(params, theta)# Set the weights for the model
    nll = torch.nn.CrossEntropyLoss(reduction='sum')(model(X), y)
    
    return -nll

def log_prior_pdf(z, prior_mean=torch.tensor(0), prior_var=torch.tensor(1)):
    """ Log prior for the weights (assuming Gaussian prior)"""
    log_prior = torch.sum(log_npdf(z, prior_mean, prior_var))
    return log_prior

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


