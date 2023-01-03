# Standard imports
import torch as tc
from graphlib import TopologicalSorter # NOTE: This is useful

# Project imports
# from evaluation_based_sampling import evaluate
from evaluate_vi import evaluate_vi
from primitives import primitives


class graph:
    def __init__(self, graph_json):
        self.json = graph_json
        self.functions = graph_json[0]
        self.graph_spec = graph_json[1]
        self.program = graph_json[-1]


def add_functions(j):
    rho = {}
    for fname, f in j.items():
        fvars = f[1]
        fexpr = f[2]
        rho[fname] = [fexpr, fvars]
    return rho


def add_graphs(j, graph_dict, sigma_dict):

    graph_vars = j["V"]
    graph_links = j["P"]
    graph_vertices = j["A"]

    ### If there is no link functino, then return empty graph dict
    if len(graph_links) == 0:
        return graph_dict, sigma_dict

    ### Sort by edges
    sorted = TopologicalSorter(graph_vertices)
    eval_order = tuple(reversed(tuple(sorted.static_order())))


    ### If there is only 1 link function, no edges
    if len(eval_order) == 0:
        single_node = list(graph_links.keys())[0]
        single_expr = list(graph_links.values())[0]

        eval_link = evaluate_vi(single_expr, rho= graph_dict, sigma=sigma_dict)

        graph_dict[single_node] = eval_link[0]
        sigma_dict['logW'] += eval_link[1]['logW']

        return graph_dict, sigma_dict

    ### Else evaluate all links in their topological order
    for gs in eval_order:
        eval_link = evaluate_vi(graph_links[gs], rho= graph_dict, sigma={'logW':0})
        graph_dict[gs] = eval_link[0]
        sigma_dict['logW'] = sigma_dict['logW'] + eval_link[1]['logW']

        # print(str(gs) + ' and ' + str(eval_link[0]) + ' and ' +  str(eval_link[1]['logW']))

    return graph_dict, sigma_dict



def Evaluate_g(j, rho, l={}, sigma=0):
    return evaluate_vi(j, l, rho, sigma)



def evaluate_graph(graph, verbose=False):
    ### Initiate environment, initiate sigma
    env = add_functions(graph.functions)
    env = {**env, **primitives}
    sig_dict = {'logW':0}

    g_dict, sig_dict = add_graphs(graph.graph_spec, env, sig_dict)

    result, sigma = Evaluate_g(graph.program, rho = g_dict, sigma=sig_dict)

    return result, sigma, None
