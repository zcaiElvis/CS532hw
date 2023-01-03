# Standard imports
import numpy as np
import torch as tc
from time import time
import wandb
import hydra

# Project imports
from daphne import load_program
from tests import is_tol, run_probabilistic_test, load_truth
from evaluate_vi import abstract_syntax_tree, evaluate_vi
from IS import Importance_sampling
from graph_based_sampling import graph
from general_sampling import get_sample
from utils import *
# from general_sampling import get_sample, prior_samples
# from evaluation_based_sampling import abstract_syntax_tree
# from graph_based_sampling import graph
# from utils import wandb_plots_homework3

# Method import
from VI import VI_sampling


def create_class(ast_or_graph, mode):
    if mode == 'desugar':
        return abstract_syntax_tree(ast_or_graph)
    elif mode == 'graph':
        return graph(ast_or_graph)
    else:
        raise ValueError('Input type not recognised')


def run_tests(tests, mode, test_type, base_dir, daphne_dir, num_samples=int(1e4), max_p_value=1e-4, compile=False, verbose=False,):

    # File paths
    test_dir = base_dir+'/programs/tests/'+test_type+'/'
    daphne_test = lambda i: test_dir+'test_%d.daphne'%(i)
    json_test = lambda i: test_dir+'test_%d_%s.json'%(i, mode)
    truth_file = lambda i: test_dir+'test_%d.truth'%(i)

    # Loop over tests
    print('Running '+test_type+' tests')
    for i in tests:
        print('Test %d starting'%i)
        print('Evaluation scheme:', mode)
        ast_or_graph = load_program(daphne_dir, daphne_test(i), json_test(i), mode=mode, compile=compile)
        ast_or_graph = create_class(ast_or_graph, mode)
        truth = load_truth(truth_file(i))
        if verbose: print('Test truth:', truth)
        if test_type == 'deterministic':
            ret, _ = get_sample(ast_or_graph, mode, verbose=verbose)
            if verbose: print('Test result:', ret)
            try:
                assert(is_tol(ret, truth))
            except AssertionError:
                raise AssertionError('Return value {} is not equal to truth {} for exp {}'.format(ret, truth, ast_or_graph))
        elif test_type == 'probabilistic':
            samples = []
            for _ in range(num_samples):
                sample, _ = get_sample(ast_or_graph, mode, verbose=verbose)
                samples.append(sample)
            p_val = run_probabilistic_test(samples, truth)
            print('p value:', p_val)
            assert(p_val > max_p_value)
        else:
            raise ValueError('Test type not recognised')
        print('Test %d passed'%i, '\n')
    print('All '+test_type+' tests passed\n')


def run_programs(programs, mode, method, prog_set, base_dir, daphne_dir, num_samples=int(1e3), tmax=None, compile=False, wandb_run=False, verbose=False,):

    # File paths
    prog_dir = base_dir+'/programs/'+prog_set+'/'
    daphne_prog = lambda i: prog_dir+'%d.daphne'%(i)
    json_prog = lambda i: prog_dir+'%d_%s.json'%(i, mode)
    results_file = lambda i: 'data/%s/%d_%s.dat'%(prog_set, i, mode)

    for i in programs:

        # Draw samples
        t_start = time()
        wandb_name = 'Program %s '%i if wandb_run else None
        print('Running: '+prog_set+':' ,i)
        print('Maximum samples [log10]:', np.log10(num_samples))
        print('Maximum time [s]:', tmax)
        print('Evaluation scheme:', mode)
        ast_or_graph = load_program(daphne_dir, daphne_prog(i), json_prog(i), mode=mode, compile=compile)
        ast_or_graph = create_class(ast_or_graph, mode)
        # samples, sigmas = prior_samples(ast_or_graph, mode, num_samples, tmax=tmax, wandb_name=wandb_name, verbose=verbose)
        # if method == "IS":
        #     samples, sigmas = Importance_sampling(ast_or_graph, mode, num_samples, tmax=tmax, wandb_name=wandb_name, verbose=verbose)
        # elif method == "MHG":
        #     samples, sigmas = Metropolis_in_Gibbs_sampler(ast_or_graph, mode, num_samples, tmax=tmax, wandb_name=wandb_name, verbose=verbose)
        # elif method == "HMC":
        #     samples, sigmas = Hamiltonian_MCMC(ast_or_graph, mode, num_samples, tmax=tmax, wandb_name=wandb_name, verbose=verbose)

        samples, _ = VI_sampling(ast_or_graph, num_samples, program=i, tmax=tmax, wandb_name=wandb_name, verbose=verbose)

        if i == 4:
            for j in range(len(samples)):
                samples[j][2] = tc.reshape(samples[j][2], (100,1))
                samples[j] = tc.cat((samples[j][0], samples[j][1], samples[j][2], samples[j][3], samples[j][4]), dim=0)




        samples = tc.stack(samples).type(tc.float)
        # samples = [tc.stack(sample).type(tc.float) for sample in samples]
        # np.savetxt(results_file(i), samples)

        # Calculate some properties of the data
        print('Samples shape:', samples.shape)
        print('First sample:', samples[0])
        print('Sample mean:', samples.mean(axis=0))
        print('Sample standard deviation:', samples.std(axis=0))
        
        if i == 2:
            print("Sample Covariance", np.cov(samples[:,0], samples[:,1]))

        # Weights & biases plots
        if wandb_run: wandb_plots_homework4(samples, i)

        # Finish
        t_finish = time()
        print('Time taken [s]:', t_finish-t_start)
        print('Number of samples:', len(samples))
        print('Finished program {}\n'.format(i))


@hydra.main(version_base=None, config_path='', config_name='config')
def run_all(cfg):

    # Configuration
    wandb_run = cfg['wandb_run']
    mode = cfg['mode']
    num_samples = int(cfg['num_samples'])
    compile = cfg['compile']
    base_dir = cfg['base_dir']
    daphne_dir = cfg['daphne_dir']
    method = cfg['method']

    # Initialize W&B
    if wandb_run: wandb.init(project='HW4', entity='cs532-2022')

    # # Deterministic tests
    # tests = cfg['deterministic_tests']
    # run_tests(tests, mode=mode, test_type='deterministic', base_dir=base_dir, daphne_dir=daphne_dir, compile=compile)

    # # Proababilistic tests
    # tests = cfg['probabilistic_tests']
    # run_tests(tests, mode=mode, test_type='probabilistic', base_dir=base_dir, daphne_dir=daphne_dir, compile=compile)

    # Programs
    programs = cfg['HW4_programs']
    run_programs(programs, mode=mode, method = method, prog_set='homework_4', base_dir=base_dir, daphne_dir=daphne_dir, num_samples=num_samples,
        compile=compile, wandb_run=wandb_run, verbose=False)



    # Finalize W&B
    if wandb_run: wandb.finish()

if __name__ == '__main__':
    run_all()
