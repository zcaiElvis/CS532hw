import torch as tc
from time import time
import numpy as np

# Project imports
from graph_based_sampling import evaluate_graph
from general_sampling import get_sample, flatten_sample
from utils import log_sample


def Importance_sampling(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):
    '''
    Importance sampling
    '''
    samples = []
    sigmas = []
    if (tmax is not None): max_time = time()+tmax

    ### Append resulting sigmas
    for i in range(num_samples):
        sample, sigma = get_sample(ast_or_graph, mode, verbose)
        samples.append(sample)
        sigmas.append(sigma["logW"])
        if type(sample) == bool: sample = tc.tensor(int(sample))
        # log_sample(sample, i, wandb_name)
        if (tmax is not None) and time() > max_time: break

    ### Log weight to weight, then normalize
    sigmas = tc.exp(tc.tensor(sigmas))
    normalized_sigmas = tc.div(tc.tensor(sigmas), sum(sigmas))

    ### Resample based on weights
    idx = tc.distributions.Categorical(normalized_sigmas).sample(tc.Size([num_samples]))
    resamples = [samples[int(x)] for x in idx]

    ### If they are boolean, make them into numbers
    if type(resamples[1]) == bool:
        resamples = [tc.tensor(int(x)) for x in resamples]

    return resamples, sigmas
