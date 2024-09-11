import autograd.numpy as np
from torch.autograd import grad
import torch
import torch.optim as optim
from time import time

log_npdf = lambda x, m, v: -(x - m) ** 2 / (2 * v) - 0.5 * torch.log(2 * torch.pi * v)

class VariationalInference(object):
    
    def __init__(self, num_params, step_size=1e-2, max_itt=2000, verbose=False, name='VariationalInference'):
        
        self.name = name
        self.verbose = verbose        
        self.X, self.y, self.ELBO = None, None, None
    
        # optimization settings
        self.num_params, self.step_size, self.max_itt = num_params, step_size, max_itt
        
        # number of parameters to be optimized is 2*D
        self.num_var_params = 2*self.num_params
        
        # Initialize the variational parameters for the mean-field approximation
        self.m, self.v = torch.zeros(num_params), torch.ones(num_params)/self.num_params
        self.lam = self.pack(self.m, self.v) # combine m and v into one vector
        
        # prepare optimizer and gradient
        self.optimizer = optim.Adam(params=[self.lam], lr=step_size)
        #(initial_param=self.lam, num_params=self.num_var_params, step_size=self.step_size)

        #self.compute_ELBO_gradient = grad(self.compute_ELBO)
        self.compute_ELBO_gradient = lambda lam: grad(self.compute_ELBO)(lam)
        
    def pack(self, m, v):
        """ Pack all parameters into one big vector for (unconstrained) optimization as follows: lam = [m, log v] """
        # Ensure m and v are PyTorch tensors
        m = torch.tensor(m, dtype=torch.float32)
        v = torch.tensor(v, dtype=torch.float32)
        
        lam = torch.zeros(self.num_var_params, dtype=torch.float32)
        lam[:self.num_params] = m
        lam[self.num_params:2*self.num_params] = torch.log(v)
        return lam
    
    def unpack(self, lam):
        """ Unpack to mean and variance from lam = [m, log-v] """
        # Ensure lam is a PyTorch tensor
        lam = torch.tensor(lam, dtype=torch.float32)
        mean = lam[:self.num_params]
        var = torch.exp(lam[self.num_params:2*self.num_params])
        return mean, var
        
    def compute_entropy(self, v=None):
        """ Compute entropy term """
        if v is None:
            # Convert self.lam to a PyTorch tensor if it's not already
            lam = torch.tensor(self.lam, dtype=torch.float32)
            v = torch.exp(lam[self.num_params:2*self.num_params])
        else:
            # Ensure v is a PyTorch tensor
            v = torch.tensor(v, dtype=torch.float32)
        entropy = 0.5 * torch.log(2 * torch.pi * v) + 0.5
        return entropy.sum()  

    def generate_posterior_samples(self, num_samples=1000):
        mean = torch.tensor(self.m, dtype=torch.float32)
        var = torch.tensor(self.v, dtype=torch.float32)
        std = torch.sqrt(var)
        samples = torch.normal(mean=mean.unsqueeze(0), std=std.unsqueeze(0).expand(num_samples, -1))
        return samples
        
    def compute_ELBO(self, lam):
        """ computes the ELBO for the linear Gaussian model based on eq. (14).
            
            Input:
            lam    -- np.array of variational parameters [m, log(v)]

            Output:
            ELBO   -- the ELBO value (scalar) for the specified variational parameters
              
             """
        
        # unpack parameters from lambda-vector
        m, v = self.unpack(lam)
        
        # implement model
        sigma2, kappa2 = 20., 10.
        expected_log_lik = torch.sum(log_npdf(self.y, self.X@m, sigma2)) - 1/(2*sigma2)*torch.sum((self.X**2)@v) 
        expected_log_prior = torch.sum(log_npdf(m, 0, kappa2) - 1/(2*kappa2)*v)                               
        entropy = self.compute_entropy(v)

        return expected_log_lik + expected_log_prior + entropy

    def fit(self, X, y, seed=0):
        """ fits the variational approximation q given data (X,y) by maximizing the ELBO using gradient-based methods """ 
        torch.manual_seed(seed)
        self.X, self.y, self.N = X, y, len(X)
        self.ELBO_history, self.lam_history = [], []
        
        print('Fitting approximation using %s' % self.name)        
        t0 = time()
        for itt in range(self.max_itt):
            
            # evaluate ELBO
            self.ELBO = self.compute_ELBO(self.lam)
            
            # store current values for plotting purposes
            self.ELBO_history.append(self.ELBO)
            self.lam_history.append(self.lam)

            # compute gradient of ELBO wrt. variational parameters
            g = self.compute_ELBO_gradient(self.lam)

            # take gradient step
            self.lam = self.optimizer.step(g)
            
            # verbose?
            if self.verbose:
                if (itt+1) % 250 == 0:
                    print('\tItt: %5d, ELBO = %3.2f' % (itt, torch.mean(self.ELBO_history[-250:])))
        
        t1 = time()
        print('\tOptimization done in %3.2fs\n' % (t1-t0))
        
        # track quantities through iterations for visualization purposes
        self.ELBO_history = torch.tensor(self.ELBO_history)
        self.lam_history = torch.tensor(self.lam_history)
        self.m = self.lam[:self.num_params]
        self.v = torch.exp(self.lam[self.num_params:2*self.num_params])
        self.m_history = self.lam_history[:, :self.num_params]
        self.v_history = torch.exp(self.lam_history[:, self.num_params:2*self.num_params])
            
        return self
    
class BlackBoxVariationalInference(VariationalInference):
    def __init__(self, theta_map, P, log_prior, log_lik, num_params, step_size=1e-2, max_itt=2000, num_samples=20, batch_size=None, seed=0, verbose=False):
    
        # arguments specific to BBVI
        self.log_prior = log_prior          # function for evaluating the log prior
        self.log_lik = log_lik              # function for evaluating the log likelihood
        self.num_samples = num_samples      # number of MC samples to use for estimation
        self.batch_size = batch_size        # batch size
        self.seed = seed                    # seed
        self.P = P                          # random projection matrix
        self.theta_map = theta_map          # MAP estimate
        
        # pass remaining argument to VI class
        super(BlackBoxVariationalInference, self).__init__(name="Black-box VI", num_params=num_params, step_size=step_size, max_itt=max_itt, verbose=verbose)
    
    def compute_ELBO(self, lam):
        """ compute an estimate of the ELBO for variational parameters in the *lam*-variable using Monte Carlo samples and the reparametrization trick.
            
            If self.batch_size is None, the function computes the ELBO using full dataset. If self.batch is not None, the function estimates the ELBO using a single minibatch of size self.batch_size. 
            The samples in the mini batch is sampled uniformly from the full dataset.
            
            inputs:
            lam       --     vector of variational parameters (unconstrained space)

            outputs:
            ELBO      --     estimate of the ELBO (scalar)

            """
        
        # unpack variational parameters
        m, v  = self.unpack(lam)
        
        # generate samples from epsilon ~ N(0, 1) and use re-parametrization trick
        epsilon = torch.randn(self.num_samples, self.num_params)

        # Compute z_samples using PyTorch tensor operations
        z_samples = m + torch.sqrt(v) * epsilon  # shape: num_samples x K

        #print('shapes:', z_samples.shape, self.P.T.shape, self.theta_map.shape)
        w_samples = z_samples @ self.P.T + self.theta_map   # shape: num_samples x D
        #w_samples -= - z_samples @ self.P.T

        # prior term (scalar )
        expected_log_prior_term = torch.mean(self.log_prior(z_samples, m, v))  # shape: scalar
    
        # batch mode or minibatching?
        if self.batch_size:
            # Use mini-batching
            
            batch_idx = torch.randperm(self.N)[:self.batch_size]

            # Use the indices to create batches
            X_batch = self.X[batch_idx]
            y_batch = self.y[batch_idx]
                    
            expected_log_lik_term = self.N/self.batch_size*self.log_lik(X_batch, y_batch, w_samples)  # shape: scalar
        else:
            # No mini-batching
            expected_log_lik_term = self.log_lik(self.X, self.y, w_samples)   # shape: scalar
        
        # compute ELBO
        ELBO = expected_log_lik_term + expected_log_prior_term + self.compute_entropy(v)
        
        return ELBO
    

