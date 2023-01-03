# Standard imports
import torch as tc
from graphlib import TopologicalSorter # NOTE: This is useful

# Project imports
from evaluation_based_sampling import evaluate
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


def add_graphs(j):
    graph_vars = j["V"]
    graph_links = j["P"]
    graph_vertices = j["A"]
    sorted = TopologicalSorter(graph_vertices)
    print(tuple(sorted.static_order()))
    return graph_links


def Evaluate_g(j, rho, l={}, sigma=0):
    return evaluate(j, l, rho, sigma)



def evaluate_graph(graph, verbose=False):
    env = add_functions(graph.functions)
    env = {**env, **primitives}
    g_dict = add_graphs(graph.graph_spec)

    env = {**env, **g_dict}

    result, sigma = Evaluate_g(graph.program, rho = env, sigma=0)

    return result, sigma, None
