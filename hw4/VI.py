from primitives import primitives
from utils import *
from distributions import *
from evaluate_vi import evaluate_vi
import torch as tc
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from distributions import *
import copy
import math
import matplotlib
import matplotlib.pyplot as plt
import pickle


from graphlib import TopologicalSorter



class q():
    def __init__(self, latent, distribution = "normal"):
        self.latent = latent
        self.distribution = distribution

        if self.distribution == "normal":
            self.q_dists = dict(zip(latent, [Normal(tc.tensor(0.),tc.tensor(1.)) for var in latent]))
    
    def sample(self):
        return [self.q_dists[var].sample() for var in self.latent]

    def show(self):
        return self.q_dists

    def show_list(self):
        return []

    def update(self, means, vars): 
        for i in range(0, len(self.latent)):
            self.q_dists[self.latent[i]] = Normal(means[i], vars[i])
        return

def add_functions(j):
    rho = {}
    for fname, f in j.items():
        fvars = f[1]
        fexpr = f[2]
        rho[fname] = [fexpr, fvars]
    return rho



def add_graphs_vi(j, graph_dict, sigma_dict, num_samples, program, wandb_name):
    graph_vars = j["V"]
    graph_links = j["P"]
    graph_vertices = j["A"]
    graph_observed = j["Y"]

    ### If there is no link functino, then return empty graph dict
    if len(graph_links) == 0:
        return graph_dict, sigma_dict

    ### Sort by edges
    sorted = TopologicalSorter(graph_vertices)
    eval_order = tuple(reversed(tuple(sorted.static_order())))

    ### Split latent and observed
    vx_observed = list(graph_observed.keys())
    vx_latent = [x for x in eval_order if x not in vx_observed]

    ### Base run
    X0 = {}
    temp_dict = copy.deepcopy(graph_dict)
    for gs in eval_order:
        eval_link = evaluate_vi(graph_links[gs], rho= temp_dict, sigma={'logW':0})
        temp_dict[gs] = eval_link[0]

        if gs in vx_latent:
            X0[gs] = eval_link[0]



    ### Constucting the Q dictionaries
    # Q_loc = [tc.clone(prior_loc) for x in vx_latent]
    # Q_scale = [tc.clone(prior_scale) for x in vx_latent]
    # Q = dict(zip(vx_latent, [Normal(Q_loc[i], Q_scale[i]) for i in range(0, len(vx_latent))]))


    # Q = {}
    # for x in vx_latent:
    #     d, _ = evaluate_vi(graph_links[x][1][1], rho={**graph_dict}, sigma={"logW":0})

    #     match type(d):

    #         case "Normal":
    #             Q[x] = Normal(tc.tensor(0.), tc.tensor(1.))

    #         case "Dirichlet":
    #             Q[x] = Dirichlet(tc.tensor([1.,1.,1.]))

    #         case "Gamma":
    #             Q[x] = Gamma(tc.tensor(1.), tc.tensor(1.))

    #         case "Discrete":
    #             Q[x] = Categorical(tc.tensor([0.3,0.3,0.4]))



    Q = {}
    for x in vx_latent:
        d, _ = evaluate_vi(graph_links[x][1], rho={**graph_dict, **X0}, sigma={"logW":0})

        match type(d):

            case tc.distributions.normal.Normal:
                Q[x] = Normal(tc.tensor(0.), tc.tensor(1.))

            case tc.distributions.dirichlet.Dirichlet:
                Q[x] = Dirichlet(tc.tensor([1.,1.,1.]))

            case tc.distributions.gamma.Gamma:
                Q[x] = Gamma(tc.tensor(1.), tc.tensor(1.))

            case tc.distributions.categorical.Categorical:
                Q[x] = Categorical(tc.tensor([0.3,0.3,0.4]))

            case tc.distributions.uniform.Uniform:
                Q[x] = Gamma(tc.tensor(1.), tc.tensor(1.))


    ### Passing dictionary to Adam
    params = [Q[var].optim_params() for var in Q.keys()]
    params = [item for sublist in params for item in sublist]
    optimizer = tc.optim.Adam(params, lr=5e-2)

    sample_total = 100

    losses = []

    for iter in tqdm(range(500)):

        ### Generate samples from Q distributions
        # Q_samples = [Q[x].sample(sample_shape=(sample_total,)) for x in vx_latent]
        # Q_samples = tc.stack(Q_samples)

        Q_samples = [{x : Q[x].sample() for x in Q.keys()} for _ in range(sample_total)]
        log_Q = tc.stack([tc.stack([Q[x].log_prob(sample[x]) for x in sample.keys()]).sum() for sample in Q_samples])

        ### Calculate log_Q
        # logQ = [Q[var].log_prob(Q_samples[i,:]) for i, var in enumerate(vx_latent)]
        # logQ = tc.stack(logQ)
        # log_Q = logQ.sum(axis=0)

        log_P = []
        for s in range(sample_total):
            X_dict = Q_samples[s]
            log_likelihood = 0
            log_prior = 0
            for v in eval_order:
                try:
                    # If sample1 less than 0.01
                    d, _ = evaluate_vi(graph_links[v][1], rho = {**graph_dict, **X_dict}, sigma={'logW':0})
                    if v in graph_observed.keys():
                        # Evaluating observation with updated z: log.Prob(x|z_new)
                        log_likelihood += d.log_prob(tc.tensor(graph_observed[v]).float())
                    else:
                        # Evaluating z with prior of z: log.Prob(z_new|z_prior) 
                        log_prior += d.log_prob(X_dict[v])
                except:
                    log_likelihood = 1e-5
                    log_prior = 1e-5

            
            log_P.append(log_likelihood + log_prior)

        log_P = tc.stack(log_P)
        
        log_W = log_P-log_Q

        ELBO_loss = -(log_Q*(log_W.detach())).mean()
        ELBO_loss.backward()
        
        ### plotting and wandb
        losses.append(ELBO_loss.clone().detach())
        log_params(Q, iter, wandb_name)
        log_loss(ELBO_loss.clone().detach(), iter, program, wandb_name)
        ###
    

        optimizer.step()
        optimizer.zero_grad()

    # print(Q)
    np.savetxt("results/Losses_{}.dat".format(program), losses)
    with open("results/Q_{}.pkl".format(program), "wb") as f:
        pickle.dump(Q, f)

    # Make sample into [{"sample1":2, "sample2":3, "sample4":[4,5]}, {"sample1":2, "sample2":3, "sample4":[4,5]}]
    resamples = [dict(zip(vx_latent, [d.sample() for d in Q.values()])) for _ in range(num_samples)]

    # Also evaluate the observed for each sample
    for i in range(num_samples):
        X_dict = resamples[i]
        ys = dict(zip(vx_observed, [evaluate_vi(graph_links[y], rho={**graph_dict, **X_dict}, sigma={'logW':0})[0] for y in vx_observed]))
        resamples[i].update(ys)

    return resamples, vx_latent, losses


def VI_sampling(ast_or_graph, num_samples, program, tmax=None, wandb_name=None, verbose=False):

    env = add_functions(ast_or_graph.functions)
    env = {**env, **primitives}
    sig_dict = {'logW':0}
    resamples, vx_latent, losses = add_graphs_vi(ast_or_graph.graph_spec, env, sig_dict, num_samples, program, wandb_name)


    results = []
    for i in range(num_samples):
        env.update(resamples[i])
        result, _ = evaluate_vi(ast_or_graph.program, rho = env, sigma = {'logw':0})
        if isinstance(result, bool): result = tc.tensor(int(result))
        results.append(result)

    ### Plotting and logging losses
    plt.plot(losses)
    plt.savefig("output.png")


    ###


    return results, None
