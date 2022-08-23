import os

# Import time for benchmarking
import time

# Get the essentials
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import pandas as pd

# Get interpolator for projecting eigenmodes between model levels.
from scipy.spatial import distance_matrix
from scipy.interpolate import interp2d
from scipy.linalg import eigh

# Get the good stuff.
import pymc3 as pm

import theano
import theano.tensor as tt

# Checking versions
print('Theano version: {}'.format(theano.__version__))
print('PyMC3 version: {}'.format(pm.__version__))

class Matern32:
    def __init__(self, coords, mkl, lamb):
        """
        This class sets up a random process
        on a grid and generates
        a realisation of the process, given
        parameters or a random vector.
        """
        
        # Internalise the grid and set number of vertices.
        self.coords = coords
        self.n_points = self.coords.shape[0]
        self.eigenvalues = None
        self.eigenvectors = None
        self.parameters = None
        self.random_field = None
        
        # Set some random field parameters.
        self.mkl = mkl
        self.lamb = lamb
        
        self.assemble_covariance_matrix()
        
    def assemble_covariance_matrix(self):        
        """
        This method creates a Matern 5/2 covariance matrix.
        """
        
        # Compute scaled distances.
        dist = np.sqrt(3)*distance_matrix(self.coords, self.coords)/self.lamb
        
        # Set up Matern 5/2 covariance matrix.
        self.cov =  (1 + dist) * np.exp(-dist)
    
    def plot_covariance_matrix(self):
        """
        Plot the covariance matrix.
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.cov, cmap='binary')
        plt.colorbar()
        plt.show()
    
    def compute_eigenpairs(self):
        """
        Find eigenvalues and eigenvectors using Arnoldi iteration.
        """
        eigvals, eigvecs = eigh(self.cov,
                                eigvals=(self.n_points - self.mkl,
                                         self.n_points - 1))
        
        order = np.flip(np.argsort(eigvals))
        self.eigenvalues = eigvals[order]
        self.eigenvectors = eigvecs[:, order]
      
    def generate(self, parameters=None):
        """
        Generate a random field, see
        Scarth, C., Adhikari, S., Cabral, P. H.,
        Silva, G. H. C., & Prado, A. P. do. (2019).
        Random field simulation over curved surfaces:
        Applications to computational structural mechanics.
        Computer Methods in Applied Mechanics and Engineering,
        345, 283–301. https://doi.org/10.1016/j.cma.2018.10.026
        """
        
        if parameters is None:
            self.parameters = np.random.normal(size=self.mkl)
        else:
            self.parameters = np.array(parameters).flatten()
        
        self.random_field = np.linalg.multi_dot((self.eigenvectors, 
                                                 np.sqrt(np.diag(self.eigenvalues)), 
                                                 self.parameters))

    def plot(self):
        """
        Plot the random field.
        """

        plt.figure(figsize=(12, 10))
        plt.tricontourf(self.coords[:, 0], self.coords[:, 1],
                        self.random_field,
                        levels=np.linspace(min(self.random_field), max(self.random_field), 20),
                        cmap='plasma')
        plt.colorbar()
        plt.show()


# ## Define the Gravity model and generate data
# This is a bit lengthy due to the model used in this case

# In[4]:


class Gravity:
    """
    Gravity is a class that implements a simple gravity surveying problem,
    as described in Hansen, P. C. (2010). Discrete Inverse Problems: Insight and Algorithms. 
    Society for Industrial and Applied Mathematics.
    It uses midpoint quadrature to evaluate a Fredholm integral of the first kind.
    """
    
    def __init__(self, f_function, depth, n_quad, n_data):
        
        # Set the function describing the distribution of subsurface density.
        self.f_function = f_function
        
        # Set the depth of the density (distance to the surface measurements).
        self.depth = depth
        
        # Set the quadrature degree along one dimension.
        self.n_quad = n_quad;
        
        # Set the number of data points along one dimension
        self.n_data = n_data
        
        # Set the quadrature points.
        x = np.linspace(0, 1, self.n_quad+1); self.tx = (x[1:] + x[:-1]) / 2
        y = np.linspace(0, 1, self.n_quad+1); self.ty = (y[1:] + y[:-1]) / 2
        TX, TY = np.meshgrid(self.tx, self.ty)
        
        # Set the measurement points.
        self.sx = np.linspace(0, 1, self.n_data)
        self.sy = np.linspace(0, 1, self.n_data)
        SX, SY = np.meshgrid(self.sx, self.sy)
        
        # Create coordinate vectors.
        self.T_coords = np.c_[TX.ravel(), TY.ravel(), np.zeros(self.n_quad**2)]
        self.S_coords = np.c_[SX.ravel(), SY.ravel(), self.depth*np.ones(self.n_data**2)]
        
        # Set the quadrature weights.
        self.w = 1/self.n_quad**2
        
        # Compute a distance matrix
        dist = distance_matrix(self.S_coords, self.T_coords)
        
        # Create the Fredholm kernel.
        self.K = self.w * self.depth/dist**3
        
        # Evaluate the density function on the quadrature points.
        self.f = self.f_function(TX, TY).flatten()
        
        # Compute the surface density (noiseless measurements)
        self.g = np.dot(self.K, self.f)
    
    def plot_model(self):
        
        # Plot the density and the signal.
        fig, axes = plt.subplots(1,2, figsize=(16,6))
        axes[0].set_title('Density')
        f = axes[0].imshow(self.f.reshape(self.n_quad, self.n_quad), extent=(0,1,0,1), origin='lower', cmap='plasma')
        fig.colorbar(f, ax=axes[0])
        axes[1].set_title('Signal')
        g = axes[1].imshow(self.g.reshape(self.n_data, self.n_data), extent=(0,1,0,1), origin='lower', cmap='plasma')
        fig.colorbar(g, ax=axes[1])
        plt.show()
        
    def plot_kernel(self):
        
        # Plot the kernel.
        plt.figure(figsize=(8,6))
        plt.imshow(self.K, cmap='plasma'); plt.colorbar()
        plt.show()

# This is the function, describing the subsurface density.
#def f(TX, TY):
#    f = np.sin(np.pi*TX) + np.sin(5*np.pi*TX) + np.sin(3*np.pi*TY) + TY + 1
#    f = f/f.max()
#    return f
def f(TX, TY):
    f = np.zeros(TX.shape)
    f[((TX-0.5)**2 + (TY-0.5)**2) < 0.25**2] = 1
    return f

# Set the model parameters.
depth = 0.1
n_quad = 100
n_data = 100

# Initialise a model
model_true = Gravity(f, depth, n_quad, n_data)

model_true.plot_model()

# Add noise to the data.
noise_level = 0.1
np.random.seed(123)
noise = np.random.normal(0, noise_level, n_data**2)
data = model_true.g + noise

class Gravity_Forward(Gravity):
    """
    Gravity forward is a class that implements the gravity problem,
    but computation of signal and density is delayed to the "solve"
    method, since it relied on a Gaussian Random Field to model
    the (unknown) density.
    """
    def __init__(self, depth, n_quad, n_data):
        
        # Set the depth of the density (distance to the surface measurements).
        self.depth = depth
        
        # Set the quadrature degree along one axis.
        self.n_quad = n_quad;
        
        # Set the number of data points along one axis.
        self.n_data = n_data
        
        # Set the quadrature points.
        x = np.linspace(0, 1, self.n_quad+1); self.tx = (x[1:] + x[:-1]) / 2
        y = np.linspace(0, 1, self.n_quad+1); self.ty = (y[1:] + y[:-1]) / 2
        TX, TY = np.meshgrid(self.tx, self.ty)
        
        # Set the measurement points.
        self.sx = np.linspace(0, 1, self.n_data)
        self.sy = np.linspace(0, 1, self.n_data)
        SX, SY = np.meshgrid(self.sx, self.sy)
        
        # Create coordinate vectors.
        self.T_coords = np.c_[TX.ravel(), TY.ravel(), np.zeros(self.n_quad**2)]
        self.S_coords = np.c_[SX.ravel(), SY.ravel(), self.depth*np.ones(self.n_data**2)]
        
        # Set the quadrature weights.
        self.w = 1/self.n_quad**2
        
        # Compute a distance matrix
        dist = distance_matrix(self.S_coords, self.T_coords)
        
        # Create the Fremholm kernel.
        self.K = self.w * self.depth/dist**3
        
    def set_random_process(self, random_process, lamb, mkl):
        
        # Set covariance length scale
        self.lamb = lamb
        
        # Set the number of KL modes.
        self.mkl = mkl
        
        # Initialise a random process on the quadrature points.
        # and compute the eigenpairs of the covariance matrix,
        self.random_process = random_process(self.T_coords, self.mkl, self.lamb)
        self.random_process.compute_eigenpairs()
    
    def solve(self, parameters):
        
        # Internalise the Random Field parameters
        self.parameters = parameters
        
        # Create a realisation of the random process, given the parameters.
        self.random_process.generate(self.parameters)
        mean = 0.0; stdev = 1.0;
        
        # Set the density.
        self.f = mean + stdev*self.random_process.random_field
        
        # Compute the signal.
        self.g = np.dot(self.K, self.f)
        
    def get_data(self):
        
        # Get the data vector.
        return self.g

# We project the eigenmodes of the fine model to the quadrature points
# of the coarse model using linear interpolation.
def project_eigenmodes(model_coarse, model_fine):
    model_coarse.random_process.eigenvalues = model_fine.random_process.eigenvalues 
    for i in range(model_coarse.mkl):
        interpolator = interp2d(model_fine.tx, model_fine.ty, model_fine.random_process.eigenvectors[:,i].reshape(model_fine.n_quad, model_fine.n_quad))
        model_coarse.random_process.eigenvectors[:,i] = interpolator(model_coarse.tx, model_coarse.ty).ravel()

# Set random process parameters.
#lamb = 0.3
#mkl = 12
lamb = 0.2
mkl = 32

# Set the quadrature degree for each model level (coarsest first)
n_quadrature = [20, 100]

# Initialise the models, according the quadrature degree.
my_models = []
for i, n_quad in enumerate(n_quadrature):
     my_models.append(Gravity_Forward(depth, n_quad, n_data))
     my_models[i].set_random_process(Matern32, lamb, mkl)
    
# Project the eigenmodes of the fine model to the coarse model.
for m in my_models[:-1]:
    project_eigenmodes(m, my_models[-1])


# Plot the same random realisation for each level, and the corresponding signal,
# to validate that the levels are equivalents.
for i, m in enumerate(my_models):
    print('Level {}:'.format(i))
    np.random.seed(123)
    m.solve(np.random.normal(size=mkl))
    m.plot_model()

plt.rcParams.update({'font.size': 14})

# Plot the density and the signal.
fig, axes = plt.subplots(1,2, figsize=(16,6))

axes[0].set_title('True Density', fontsize=16)
f = axes[0].imshow(model_true.f.reshape(model_true.n_quad, model_true.n_quad), extent=(0,1,0,1), origin='lower', cmap='plasma')
fig.colorbar(f, ax=axes[0])
axes[0].grid(False)

axes[1].set_title('Noisy Signal', fontsize=16)
d = axes[1].imshow(data.reshape(n_data, n_data), extent=(0,1,0,1), origin='lower', cmap='plasma')
fig.colorbar(d, ax=axes[1])
axes[1].grid(False)
plt.savefig('true_density_noisy_signal.png', dpi=300)


np.random.seed(123)
random_parameters = np.random.normal(size=mkl)
for m in my_models:
    m.solve(random_parameters)

# Plot the density and the signal.
fig, axes = plt.subplots(1,2, figsize=(16,6))

axes[0].set_title('Coarse Model', fontsize=16)
f = axes[0].imshow(my_models[0].f.reshape(my_models[0].n_quad, my_models[0].n_quad), extent=(0,1,0,1), origin='lower', cmap='plasma')
fig.colorbar(f, ax=axes[0])
axes[0].grid(False)

axes[1].set_title('Fine Model', fontsize=16)
d = axes[1].imshow(my_models[1].f.reshape(my_models[1].n_quad, my_models[1].n_quad), extent=(0,1,0,1), origin='lower', cmap='plasma')
fig.colorbar(d, ax=axes[1])
axes[1].grid(False)

plt.savefig('coarse_and_fine_model.png', dpi=300)


# Number of draws from the distribution
ndraws = 20000

# Number of burn-in samples
nburn = 5000

# MLDA and Metropolis tuning parameters
tune = True
tune_interval = 100
discard_tuning = True
scaling = 0.01

base_sampler = 'Metropolis'

# Number of independent chains. 
nchains = 4

# Subsampling rate for MLDA
nsub = 10

# Do blocked/compounds sampling in Metropolis and MLDA 
# Note: This choice applies only to the coarsest level in MLDA 
# (where a Metropolis sampler is used), all other levels use block sampling
blocked = True

# Set prior parameters for multivariate Gaussian prior distribution.
mu_prior = np.zeros(mkl)
cov_prior = np.eye(mkl)

# Set the sigma for inference.
sigma = 1.0

# Data generation seed
data_seed = 1234

# Sampling seed
sampling_seed = 1234


def my_loglik(my_model, theta, data, sigma):
    """
    This returns the log-likelihood of my_model given theta,
    datapoints, the observed data and sigma. It uses the
    model_wrapper function to do a model solve.
    """
    my_model.solve(theta)
    output = my_model.get_data()
    return - (0.5 / sigma ** 2) * np.sum((output - data) ** 2)

class LogLike(tt.Op):
    """
    Theano Op that wraps the log-likelihood computation, necessary to
    pass "black-box" code into pymc3.
    Based on the work in:
    https://docs.pymc.io/notebooks/blackbox_external_likelihood.html
    https://docs.pymc.io/Advanced_usage_of_Theano_in_PyMC3.html
    """

    # Specify what type of object will be passed and returned to the Op when it is
    # called. In our case we will be passing it a vector of values (the parameters
    # that define our model and a model object) and returning a single "scalar"
    # value (the log-likelihood)
    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)
    
    def __init__(self, my_model, loglike, data, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        my_model:
            A Model object (defined in model.py) that contains the parameters
            and functions of out model.
        loglike:
            The log-likelihood function we've defined, in this example it is
            my_loglik.
        data:
            The "observed" data that our log-likelihood function takes in. These
            are the true data generated by the finest model in this example.
        x:
            The dependent variable (aka 'x') that our model requires. This is
            the datapoints in this example.
        sigma:
            The noise standard deviation that our function requires.
        """
        # add inputs as class attributes
        self.my_model = my_model
        self.likelihood = loglike
        self.data = data
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(self.my_model, theta, self.data, self.sigma)

        outputs[0][0] = np.array(logl) # output the log-likelihood


# create Theano Ops to wrap likelihoods of all model levels and store them in list
logl = []
for i, m_i in enumerate(my_models):
    logl.append(LogLike(m_i, my_loglik, data, sigma))


# ## Create coarse model in PyMC3

# Set up models in pymc3 for each level - excluding finest model level
coarse_models = []
for j in range(len(my_models) - 1):
    with pm.Model() as model:
        
        # Multivariate normal prior.
        theta = pm.MvNormal('theta', mu=mu_prior, cov=cov_prior, shape=mkl)        
        
        # Use the Potential class to evaluate likelihood
        pm.Potential('likelihood', logl[j](theta))
        
    coarse_models.append(model)


# ## Create fine model and perform inference

# Set up finest model and perform inference with PyMC3, using the MLDA algorithm
# and passing the coarse_models list created above.
method_names = []
traces = []
runtimes = []

with pm.Model() as model:
    
    # Multivariate normal prior.
    theta = pm.MvNormal('theta', mu=mu_prior, cov=cov_prior, shape=mkl)
    
    # Use the Potential class to evaluate likelihood
    pm.Potential('likelihood', logl[-1](theta))
    
    # Find the MAP estimate for convergence diagnostics. 
    # It is NOT used as starting point for sampling.
    MAP = pm.find_MAP() 
    
    # Initialise Metropolis step method
    step_metropolis = pm.Metropolis(tune=tune, tune_interval=tune_interval, blocked=blocked, scaling=scaling)
    
    # Sample Metropolis
    t_start = time.time()
    method_names.append("Metropolis")
    traces.append(pm.sample(draws=ndraws, step=step_metropolis,
                            chains=nchains, tune=nburn,
                            discard_tuned_samples=discard_tuning,
                            random_seed=sampling_seed, start=MAP, 
                            cores=4))
    runtimes.append(time.time() - t_start)
    
    # Initialise MLDA step method    
    step_mlda = pm.MLDA(coarse_models=coarse_models, subsampling_rates=nsub, 
                        tune=tune, base_tune_interval=tune_interval, 
                        base_blocked=blocked, base_scaling=scaling,
                        base_sampler=base_sampler)

    # Sample MLDA
    t_start = time.time()
    method_names.append("MLDA")
    traces.append(pm.sample(draws=ndraws, step=step_mlda,
                            chains=nchains, tune=nburn,
                            discard_tuned_samples=discard_tuning,
                            random_seed=sampling_seed, start=MAP, 
                            cores=4))
    runtimes.append(time.time() - t_start)   


# ## Get post-sampling stats and diagnostics 

# #### Print MAP estimate and pymc3 sampling summary

with model:
    print(f"\nDetailed summaries and plots:\nMAP estimate: {MAP['theta']}.")
    for i, trace in enumerate(traces):
        print(f"\n{method_names[i]} Sampler:\n") 
        print(pm.stats.summary(trace))

acc = []
ess = []
ess_n = []
performances = []

# Get some more statistics.
with model:
    for i, trace in enumerate(traces):
        acc.append(trace.get_sampler_stats('accepted').mean())
        ess.append(np.array(pm.ess(trace).to_array()))
        ess_n.append(ess[i] / len(trace) / trace.nchains)
        performances.append(ess[i] / runtimes[i])
        print(f'\n{method_names[i]} Sampler: {len(trace)} drawn samples in each of '
              f'{trace.nchains} chains.'
              f'\nRuntime: {runtimes[i]} seconds'
              f'\nAcceptance rate: {acc[i]}'
              f'\nESS list: {np.round(ess[i][0], 3)}'
              f'\nNormalised ESS list: {np.round(ess_n[i][0], 3)}'
              f'\nES/sec: {np.round(performances[i][0], 3)}')

import pickle

gravity_metropolis = {'acceptance': acc[0], 'ess': ess[0], 'ess_normalised': ess_n[0], 'performance': performances[0], 'trace': traces[0].get_values('theta')}
gravity_mlda = {'acceptance': acc[1], 'ess': ess[1], 'ess_normalised': ess_n[1], 'performance': performances[1], 'trace': traces[1].get_values('theta')}

with open('gravity_metropolis.p', 'wb') as file:
    pickle.dump(gravity_metropolis, file)
    
with open('gravity_mlda.p', 'wb') as file:
    pickle.dump(gravity_mlda, file)


