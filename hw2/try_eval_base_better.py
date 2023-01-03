# Standard imports
import torch as tc
import json

# Project imports
from primitives import primitives # NOTE: Import and use this!



class abstract_syntax_tree:
    def __init__(self, ast_json):
        self.ast_json = ast_json
        self.functions = ast_json[:-1]
        self.program = ast_json[-1]


def add_functions(j):
    rho = {}
    for fun in j:
        fname = j[0][1]
        fvars = j[0][2]
        fexpr = j[0][3]
        rho[fname] = [fexpr, fvars]
    return rho


def evaluate(j, e, rho, sigma):
    match e:
        case isinstance(e, bool):
            return tc.tensor(int(j)).float()


# def evaluate(j, l={}, rho={}, sigma={}):
#
#     ### Branch: True/False, return 1/0
#     if isinstance(j, bool):
#         return tc.tensor(int(j)).float()
#
#     ### Branch: If int/float, return tensor version
#     elif isinstance(j, int) or isinstance(j, float):
#         return tc.tensor(j).float()
#
#     ### If observe, update weight sigma
#     elif j[0] == "observe":
#         d = evaluate(j[1], l = l, rho = rho, sigma = sigma)
#         v = evaluate(j[2], l=l, rho = rho, sigma = sigma)
#         logp = d.log_prob(v)
#         sigma['logW']+= tc.tensor(logp)
#         return v
#
#     ### If sample, or sample*, evaluate distribution and sample
#     elif j[0] == "sample" or j[0] == "sample*":
#         return evaluate(j[1], l = l, rho=rho, sigma = sigma).sample()
#
#     ### If: lazy evaluation
#     elif j[0] == "if":
#         true_or_not = evaluate(j[1], l = l , rho= rho, sigma= sigma)
#         if(true_or_not):
#             return evaluate(j[2], l = l, rho=rho, sigma= sigma)
#         else:
#             return evaluate(j[3], l = l, rho=rho, sigma= sigma)
#
#     ### Let: add to local variable l
#     elif j[0] == "let":
#         c = evaluate(j[1][1], l, rho, sigma= sigma)
#         l[j[1][0]] = c
#         return evaluate(j[2], l, rho, sigma= sigma)
#
#     ### If it is a string
#     elif isinstance(j, str):
#         ### If string found in environment, evaluate it
#         if j in rho:
#             return evaluate(rho[j], l=l, rho=rho, sigma= sigma)
#         ### If not, then it is a local variable
#         else:
#             return l[j]
#
#     ### Lastly, if it is a list
#     else:
#         opt = rho[j[0]]
#
#         values = []
#         for i in range(1, len(j)):
#             values.append(evaluate(j[i], l, rho, sigma= sigma))
#
#         ### If rho is a list, not operator, then it's a function dictionary
#         if isinstance(opt, list):
#             fvars = rho[j[0]][1]
#             fexpr = rho[j[0]][0]
#             localenv = dict(zip(fvars, values))
#             return evaluate(fexpr, l = localenv, rho = rho, sigma= sigma)
#
#         return opt(*values)



def evaluate_program(ast, verbose=False):

    env = add_functions(ast.functions)
    env = {**env, **primitives}
    result = evaluate(ast.program, rho = env, sigma=0)

    return result, None, None
