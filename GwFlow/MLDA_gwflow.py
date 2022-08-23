import os as os
import sys as sys
import time as time

from itertools import product

import arviz as az
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np

import pymc3 as pm
import theano.tensor as tt

import fenics as fn

sys.path.insert(1, 'mlda/') # import groundwater flow model utils (including FEniCS code)
from Model import Model, model_wrapper, project_eigenpairs

#%config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)
#az.style.use('arviz-darkgrid')

# Set the resolution of the multi-level models (from coarsest to finest)
# This is a list of different model resolutions. Each
# resolution added to the list will add one level to the multi-level
# inference. Each element is a tuple (x,y) where x, y are the number of 
# points in each dimension. For example, here we set resolutions = 
# [(4, 4), (20, 20), (80, 80)] which creates a coarse, cheap 4x4 model,
# a finer 20x20 model and the finest, expensive 80x80 model.
resolutions = [(4, 4), (16,16), (64, 64)]
n_eig = [1, 1, 64]

# Set random field parameters
field_mean = 0
field_stdev = 2
lamb_cov = 0.1

# Set the number of unknown parameters (i.e. dimension of theta in posterior)
nparam = n_eig[-1]

# Number of draws from the distribution
ndraws = 20000

# Number of burn-in samples
nburn = 5000

# MLDA tuning parameters
tune = True
tune_interval = 100
discard_tuning = True
scaling = 0.01

# Number of independent chains
nchains = 2

# Subsampling rate for MLDA
nsub = 5

# Do blocked/compounds sampling in MLDA 
# Note: This choice applies only to the coarsest level in MLDA 
# (and only when a Metropolis base sampler is used), all other levels use block sampling
blocked = True

# Set the sigma for inference
sigma = 0.1

# Data generation seed
data_seed = RANDOM_SEED

# Sampling seed
sampling_seed = RANDOM_SEED

# Datapoints list
points_list = [0.1, 0.3, 0.5, 0.7, 0.9]

class ForwardModel_AEM(tt.Op):
    """
    Theano Op that wraps the forward model computation,
    necessary to pass "black-box" fenics code into pymc3.
    
    This model is written so that it can be used in combination
    with the MLDA's adaptive error model (AEM)
    """

    # Specify what type of object will be passed and returned to the Op when it is
    # called. In our case we will be passing it a vector of values (the parameters
    # that define our model) and returning a vector of model outputs
    itypes = [tt.dvector]  # expects a vector of parameter values (theta)
    otypes = [tt.dvector]  # outputs a vector of model outputs

    def __init__(self, my_model, x, pymc3_model):
        """
        Initialise the Op with various things that our forward model function
        requires.

        Parameters
        ----------
        my_model:
            A Model object (defined in file model.py) that contains the parameters
            and functions of our model.
        x:
            The dependent variable (aka 'x') that our model requires. This is
            the datapoints in this example.
        pymc3_model:
            The PyMC3 model being used.
        """
        # add inputs as class attributes
        self.my_model = my_model
        self.x = x
        self.pymc3_model = pymc3_model

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta = inputs[0]  # this will contain my variables

        # call the forward model function
        temp = model_wrapper(self.my_model, theta, self.x)
        
        # NOTE: This line need to be included in the perform() function is you want to use AEM
        # It saves the forward model output to a data variable (`model_output`) within the PyMC3 model
        # this allows PyMC3 to access the value during sampling
        self.pymc3_model.model_output.set_value(temp)
            
        # write to output
        outputs[0][0] = temp


class ForwardModel(tt.Op):
    """
    Theano Op that wraps the forward model computation,
    necessary to pass "black-box" fenics code into pymc3. 
    
    This is not compatible to AEM.
    """

    # Specify what type of object will be passed and returned to the Op when it is
    # called. In our case we will be passing it a vector of values (the parameters
    # that define our model) and returning a vector of model outputs
    itypes = [tt.dvector]  # expects a vector of parameter values (theta)
    otypes = [tt.dvector]  # outputs a vector of model outputs

    def __init__(self, my_model, x):
        """
        Initialise the Op with various things that our forward model function
        requires.

        Parameters
        ----------
        my_model:
            A Model object (defined in file model.py) that contains the parameters
            and functions of our model.
        x:
            The dependent variable (aka 'x') that our model requires. This is
            the datapoints in this example.        
        """
        # add inputs as class attributes
        self.my_model = my_model
        self.x = x

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        theta = inputs[0]  # this will contain my variables

        # call the forward model function
        temp = model_wrapper(self.my_model, theta, self.x)
                    
        # write to output
        outputs[0][0] = temp

# Note this can take a few minutes for large resolutions
my_models = []
for r, e in zip(resolutions, n_eig):
    my_models.append(Model(r, field_mean, field_stdev, e, lamb_cov))

# Project eignevactors from fine model to all coarse models
for i in range(len(my_models[:-1])):
    project_eigenpairs(my_models[-1], my_models[i])


# Solve finest model as a test and plot transmissivity field and solution
np.random.seed(data_seed)
my_models[-1].solve()

# Save true parameters of finest model
true_parameters = my_models[-1].random_process.parameters

my_models[0].solve(true_parameters)
my_models[1].solve(true_parameters)

plt.rcParams.update({'font.size': 14})

# Plot the density and the signal.
plt.figure(figsize=(24,6))

plt.subplot(1,3,1)
plt.title('Coarse Model', fontsize=16)
K_coarse = fn.Function(my_models[0].solver.V)
K_coarse.vector()[:] = np.log(my_models[0].solver.K.vector()[:])
f = fn.plot(K_coarse, cmap='plasma')
plt.colorbar(f)
plt.grid(False)

plt.subplot(1,3,2)
plt.title('Middle Model', fontsize=16)
K_mid = fn.Function(my_models[1].solver.V)
K_mid.vector()[:] = np.log(my_models[1].solver.K.vector()[:])
f = fn.plot(K_mid, cmap='plasma')
plt.colorbar(f)
plt.grid(False)

plt.subplot(1,3,3)
plt.title('Fine Model', fontsize=16)
K_fine = fn.Function(my_models[-1].solver.V)
K_fine.vector()[:] = np.log(my_models[-1].solver.K.vector()[:])
f = fn.plot(K_fine, cmap='plasma')
plt.colorbar(f)
plt.grid(False)

plt.savefig('coarse_and_fine_model.png', dpi=300)

plt.figure(figsize=(8,6))
plt.plot(my_models[-1].random_process.eigenvalues)
plt.savefig('eigenvalues.png', dpi=300)

# Define the sampling points.
x_data = y_data = np.array(points_list)
datapoints = np.array(list(product(x_data, y_data)))

# Get data from the sampling points and perturb it with some noise.
noise = np.random.normal(0, 0.001, len(datapoints))

# Generate data from the finest model for use in pymc3 inference - these data are used in all levels
data = model_wrapper(my_models[-1], true_parameters, datapoints) + noise

# Create covariance matrix of normal error - it is a diagonal matrix
s = np.identity(len(data))
np.fill_diagonal(s, sigma**2)


# ### Perform inference using MLDA (with AEM)
# Here we define the finest model and perform inference with PyMC3, using MLDA with AEM
# 
# Note that in the finest model definition we do not need the variables `mu_B` and `Sigma_B`, since the finest model is not corrected.

# In[27]:


coarse_models = []
mout = []
for j in range(len(my_models) - 1):
    with pm.Model() as model:
        # mu_B and Sigma_B are the mean and covariance of the bias
        # between this forward model and the model one level below. The bias is due
        # to different levels of coarseness, i.e. resolutions.
        # Both are initialised with zeros.
        # These will be updated in each iteration of this level's chain
        mu_B = pm.Data('mu_B', np.zeros(len(data)))
        Sigma_B = pm.Data('Sigma_B', np.zeros((len(data), len(data))))
        
        # This will be used to store the output of the forward model produced by the Op
        # The user needs to save the model output to this variable inside the 
        # perform() method of the Op
        model_output = pm.Data('model_output', np.zeros(len(data)))

        # Sigma_e is the covariance of the assumed error 'e' in the model.
        # This error is due to measurement noise/bias vs. the real world
        Sigma_e = pm.Data('Sigma_e', s)

        # uniform priors on unknown parameters
        parameters = []
        for i in range(nparam):
            parameters.append(pm.Normal('theta_' + str(i), 0, sigma=1.))

        # convert thetas to a tensor vector
        theta = tt.as_tensor_variable(parameters)

        # Here we instatiate a ForwardModel_AEM using the class defined above 
        # (which wraps the fenics model code) and we add to the mout list
        mout.append(ForwardModel_AEM(my_models[j], datapoints, model))
                
        # The distribution of the error 'e' (assumed error of the forward model)
        # This is multi-variate normal where:
        # - the mean is equal to the forward model output plus the bias correction term mu_B
        # - the covariance is equal to the forward model covariance Sigma_e plus the bias correction term Sigma_B
        # This creates the likelihood of the model given the observed data
        pm.MvNormal('e', mu=mout[j](theta) + mu_B, cov=Sigma_e + Sigma_B, observed=data)

    coarse_models.append(model)


# ### Construct pymc3 model objects for coarse models (without AEM)
# This defines PyMC3 models for each level - excluding finest model level.
# 
# Notice that here we do not use any of the special variables that we used without AEM.

# In[ ]:


method_names = []
traces = []
runtimes = []
acc = []
ess = []
ess_n = []
performances = []

with pm.Model() as fine_model:
    # This will be used to store the output of the forward model produced by the Op
    # The user needs to save the model output to this variable inside the 
    # perform() method of the Op
    model_output = pm.Data('model_output', np.zeros(len(data)))

    # Sigma_e is the covariance of the assumed error 'e' in the model.
    # This error is due to measurement noise/bias vs. the real world
    Sigma_e = pm.Data('Sigma_e', s)

    # uniform priors on unknown parameters
    parameters = []
    for i in range(nparam):
        parameters.append(pm.Normal('theta_' + str(i), 0, sigma=1.))

    # convert thetas to a tensor vector
    theta = tt.as_tensor_variable(parameters)

    # Here we instatiate a ForwardModel_AEM using the class defined above 
    # (which wraps the fenics model code) and we add to the mout list
    mout.append(ForwardModel_AEM(my_models[-1], datapoints, fine_model))      

    # The distribution of the error 'e' (assumed error of the forward model)
    # Note that here we *do not* correct the mean and covariance since this is the finest model
    pm.MvNormal('e', mu=mout[-1](theta), cov=Sigma_e, observed=data)
    
    MAP = pm.find_MAP() 

    # Initialise an MLDA step method object, passing the subsampling rate and
    # coarse models list and activate AEM
    step_mlda_with = pm.MLDA(subsampling_rate=nsub, coarse_models=coarse_models,
                             tune=tune, tune_interval=tune_interval, base_blocked=blocked,
                             base_scaling=scaling, adaptive_error_model=True)
    
    # inference
    t_start = time.time()
    method_names.append("MLDA_with_AEM")
    traces.append(pm.sample(draws=ndraws, step=step_mlda_with,
                            chains=nchains, tune=nburn,
                            discard_tuned_samples=discard_tuning,
                            random_seed=sampling_seed,
                            cores=1, start=MAP))
    runtimes.append(time.time() - t_start)


parameters_with = np.zeros((traces[0].get_values(traces[0].varnames[0]).shape[0], nparam))

for i in range(nparam):
    parameters_with[:,i] = traces[0].get_values(traces[0].varnames[i])

np.savetxt('parameters_with.csv', parameters_with)


# Set up models in pymc3 for each level - excluding finest model level
coarse_models = []
mout = []
for j in range(len(my_models) - 1):
    with pm.Model() as model:
        # Sigma_e is the covariance of the assumed error 'e' in the model.
        # This error is due to measurement noise/bias vs. the real world
        Sigma_e = pm.Data('Sigma_e', s)

        # uniform priors on unknown parameters
        parameters = []
        for i in range(nparam):
            parameters.append(pm.Normal('theta_' + str(i), 0, sigma=1.))

        # convert thetas to a tensor vector
        theta = tt.as_tensor_variable(parameters)

        # Here we instatiate a ForwardModel and pass the model as argument
        mout.append(ForwardModel(my_models[j], datapoints))
        
        # The distribution of the error 'e' (assumed error of the forward model)
        # This is multi-variate normal where:
        # - the mean is equal to the forward model output
        # - the covariance is equal to the forward model covariance Sigma_e
        # This creates the likelihood of the model given the observed data
        pm.MvNormal('e', mu=mout[j](theta), cov=Sigma_e, observed=data)

    coarse_models.append(model)




# Set up finest model and perform inference with PyMC3, using the MLDA algorithm
# and passing the coarse_models list created above.
with pm.Model() as fine_model:
    # Sigma_e is the covariance of the assumed error 'e' in the model.
    # This error is due to measurement noise/bias vs. the real world
    Sigma_e = pm.Data('Sigma_e', s)

    # uniform priors on unknown parameters
    parameters = []
    for i in range(nparam):
        parameters.append(pm.Normal('theta_' + str(i), 0, sigma=1.))

    # convert thetas to a tensor vector
    theta = tt.as_tensor_variable(parameters)

    # Here we instatiate a ForwardModel and pass the model as argument
    mout.append(ForwardModel(my_models[-1], datapoints))
    
    # The distribution of the error 'e' (assumed error of the forward model)
    pm.MvNormal('e', mu=mout[-1](theta), cov=Sigma_e, observed=data)

    MAP = pm.find_MAP() 
    
    # Initialise an MLDA step method object, passing the subsampling rate and
    # coarse models list    
    step_mlda_without = pm.MLDA(subsampling_rate=nsub, coarse_models=coarse_models,
                             tune=tune, tune_interval=tune_interval, base_blocked=blocked,
                             base_scaling=scaling, adaptive_error_model=False)
    
    # inference
    t_start = time.time()
    method_names.append("MLDA_without_AEM")
    traces.append(pm.sample(draws=ndraws, step=step_mlda_without,
                            chains=nchains, tune=nburn,
                            discard_tuned_samples=discard_tuning,
                            random_seed=sampling_seed,
                            cores=1, start=MAP))
    runtimes.append(time.time() - t_start)


parameters_without = np.zeros((traces[1].get_values(traces[1].varnames[0]).shape[0], nparam))

for i in range(nparam):
    parameters_without[:,i] = traces[1].get_values(traces[1].varnames[i])

np.savetxt('parameters_without.csv', parameters_without)


for i, trace in enumerate(traces):
    acc.append(trace.get_sampler_stats('accepted').mean())
    ess.append(np.array(az.ess(trace).to_array()))
    ess_n.append(ess[i] / len(trace) / trace.nchains)
    performances.append(ess[i] / runtimes[i])
    print(f'\nSampler {method_names[i]}: {len(trace)} drawn samples in each of '
          f'{trace.nchains} chains.'
          f'\nRuntime: {runtimes[i]} seconds'
          f'\nAcceptance rate: {acc[i]}'
          f'\nESS list: {ess[i]}'
          f'\nNormalised ESS list: {ess_n[i]}'
          f'\nESS/sec: {performances[i]}')

print(f"\nMLDA+AEM vs. MLDA performance speedup in all dimensions (performance measured by ES/sec):\n{np.array(performances[0]) / np.array(performances[1])}")


print(f"\nDetailed summaries and plots:\nTrue parameters: {true_parameters}")
for i, trace in enumerate(traces):
    print(f"\nSampler {method_names[i]}:\n", pm.stats.summary(trace))
