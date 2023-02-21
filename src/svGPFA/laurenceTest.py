# https://github.com/joacorapela/svGPFA code from here with some modifications

import time
import warnings
import torch
import pickle

import svGPFA.stats.kernels
import svGPFA.stats.svGPFAModelFactory
import svGPFA.stats.svEM
import svGPFA.utils.miscUtils
import svGPFA.utils.initUtils
import gcnu_common.stats.pointProcesses.tests

# Load data

# The spikes times of all neurons in all trials should be stored in nested lists.
sim_res_filename = r"C:\Users\laurence\Documents\svGPFA\examples\data\32451751_simRes.pickle" # simulation results filename
with open(sim_res_filename, "rb") as f:
    sim_res = pickle.load(f)                                                 
spikes_times = sim_res["spikes"]

#==============================================================================
                                     # Parmams   

# Hyperparameters
n_latents = 2
trials_start_time = 0.0
trials_end_time = 1.0
em_max_iter = 2

n_trials = len(spikes_times)
n_neurons = len(spikes_times[0])
trials_start_times = [trials_start_time] * n_trials
trials_end_times = [trials_end_time] * n_trials

# Get defaults
# build default parameter specificiations                                                                                                                                              
default_params_spec = svGPFA.utils.initUtils.getDefaultParamsDict(n_neurons=n_neurons, 
                                                                  n_trials=n_trials, 
                                                                  n_latents=n_latents,
                                                                  em_max_iter=em_max_iter)

# get parameters and kernels types from the parameters specification
params, kernels_types = svGPFA.utils.initUtils.getParamsAndKernelsTypes(n_trials=n_trials, 
                                                                        n_neurons=n_neurons, 
                                                                        n_latents=n_latents,
                                                                        trials_start_times=trials_start_times, 
                                                                        trials_end_times=trials_end_times,
                                                                        default_params_spec=default_params_spec)

# build kernels
kernels_params0 = params["initial_params"]["posterior_on_latents"]["kernels_matrices_store"]["kernels_params0"]
kernels = svGPFA.utils.miscUtils.buildKernels(
    kernels_types=kernels_types, kernels_params=kernels_params0)

#==============================================================================
                                        # Model

# create model
model = svGPFA.stats.svGPFAModelFactory.SVGPFAModelFactory.\
    buildModelPyTorch(kernels=kernels)

# set initial parameters
model.setParamsAndData(
    measurements=spikes_times,
    initial_params=params["initial_params"],
    eLLCalculationParams=params["ell_calculation_params"],
    priorCovRegParam=params["optim_params"]["prior_cov_reg_param"])

#==============================================================================
                                    # Optimization


# maximize lower bound
svEM = svGPFA.stats.svEM.SVEM_PyTorch()
tic = time.perf_counter()
lowerBoundHist, elapsedTimeHist, terminationInfo, iterationsModelParams = \
    svEM.maximize(model=model, optim_params=params["optim_params"],
                  method=params["optim_params"]["optim_method"])
toc = time.perf_counter()
print(f"Elapsed time {toc - tic:0.4f} seconds")

# =============================================================================
#                                  # Plotting
import numpy as np
import pandas as pd
import sklearn.metrics
import plotly.express as px
import svGPFA.plot.plotUtilsPlotly

# Plot lower bound
fig = svGPFA.plot.plotUtilsPlotly.getPlotLowerBoundHist(lowerBoundHist=lowerBoundHist)
fig.show()

# Plot latent trajectories
neuron_to_plot = 0
latent_to_plot = 0
n_time_steps_CIF = 100
trials_colorscale = "hot"

# set times to plot
trials_start_times = [trials_start_time for r in range(n_trials)]
trials_end_times = [trials_end_time for r in range(n_trials)]
trials_times = svGPFA.utils.miscUtils.getTrialsTimes(
    start_times=trials_start_times,
    end_times=trials_end_times,
    n_steps=n_time_steps_CIF)

# set trials colors
trials_colors = px.colors.sample_colorscale(
    colorscale=trials_colorscale, samplepoints=n_trials,
    colortype="rgb")
trials_colors_patterns = [f"rgba{trial_color[3:-1]}, {{:f}})" for trial_color in trials_colors]

# set trials labels
trials_labels = [str(r) for r in range(n_trials)]

# plot estimated latent across trials
test_mu_k, test_var_k = model.predictLatents(times=trials_times)
fig = svGPFA.plot.plotUtilsPlotly.getPlotLatentAcrossTrials(times=trials_times.numpy(), latentsMeans=test_mu_k, latentsSTDs=torch.sqrt(test_var_k), latentToPlot=latent_to_plot, trials_colors_patterns=trials_colors_patterns, xlabel="Time (msec)")
fig.show()

# Plot embeddings
embedding_means, embedding_vars = model.predictEmbedding(times=trials_times)
embedding_means = embedding_means.detach().numpy()
embedding_vars = embedding_vars.detach().numpy()
title = "Neuron {:d}".format(neuron_to_plot)
fig = svGPFA.plot.plotUtilsPlotly.getPlotEmbeddingAcrossTrials(times=trials_times.numpy(), embeddingsMeans=embedding_means[:,:,neuron_to_plot], embeddingsSTDs=np.sqrt(embedding_vars[:,:,neuron_to_plot]), trials_colors_patterns=trials_colors_patterns, title=title)
fig.show()
