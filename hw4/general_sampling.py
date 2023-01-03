# Standard imports
import torch as tc
from time import time
import numpy as np

# Project imports
from evaluate_vi import evaluate_program
from graph_based_sampling import evaluate_graph
# from utils import log_sample
from utils import wandb_plots_homework3

def flatten_sample(sample):
    if type(sample) is list: # NOTE: Nasty hack for the output from program 4 of homework 2
        flat_sample = tc.concat([element.flatten() for element in sample])
    else:
        flat_sample = sample
    return flat_sample


def get_sample(ast_or_graph, mode, verbose=False):
    if mode == 'desugar':
        ret, sig, _ = evaluate_program(ast_or_graph, verbose=verbose)
    elif mode == 'graph':
        ret, sig, _ = evaluate_graph(ast_or_graph, verbose=verbose)
    else:
        raise ValueError('Mode not recognised')
    ret = flatten_sample(ret)
    return ret, sig


def prior_samples(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):
    '''
    Generate a set of samples from the prior of a FOPPL program
    '''
    samples = []
    if (tmax is not None): max_time = time()+tmax
    for i in range(num_samples):
        sample, _ = get_sample(ast_or_graph, mode, verbose)
        if wandb_name is not None: log_sample(sample, i, wandb_name=wandb_name)
        samples.append(sample)
        if (tmax is not None) and time() > max_time: break
    return samples



# def prior_samples(ast_or_graph, mode, num_samples, tmax=None, wandb_name=None, verbose=False):
#     '''
#     Importance sampling
#     '''
#     samples = []
#     sigmas = []
#     if (tmax is not None): max_time = time()+tmax
#     for i in range(num_samples):
#         sample, sigma = get_sample(ast_or_graph, mode, verbose)
#         samples.append(sample)
#         sigmas.append(sigma["logW"])
#         if (tmax is not None) and time() > max_time: break
#
#
#
#     sigmas = tc.exp(tc.tensor(sigmas))
#     normalized_sigmas = tc.div(tc.tensor(sigmas), sum(sigmas))
#
#
#     idx = tc.distributions.Categorical(normalized_sigmas).sample(tc.Size([num_samples]))
#     resamples = [samples[int(x)] for x in idx]
#
#     if type(resamples[1]) == bool:
#         resamples = [tc.tensor(int(x)) for x in resamples]
#
#
#     return resamples, sigmas
