# Standard imports
import torch as tc
import json

# Project imports
from primitives import primitives

# class exp_type:
#     BOOLEAN = bool()
#     INT = int()
#     FLOAT = float()
#     LET = "let"
#     OBSERVE = "observe"
#     IF = "if"
#     SAMPLE = "sample"
#     STRING = str()
#     LIST = list()

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


# def evaluate(e, rho, sigma, l):
#
#     print(e)
#     match e:
#         case bool():
#             return tc.tensor(int(e)).float()
#
#         case int() | float():
#             return tc.tensor(e).float()
#
#         case "let":
#             c = evaluate(e[1][1], l, rho, sigma= sigma)
#             l[e[1][0]] = c
#             return evaluate(e[2], l, rho, sigma= sigma)
#
#         case list():
#             print("here")
#             opt = rho[e[0]]
#             values = []
#             for i in range(1, len(e)):
#                 values.append(evaluate(e[i], rho, sigma= sigma, l = l))
#             return opt(*values)
#
#
#
#     print("fuck, matching failed")
#     return


def evaluate(j, l={}, rho={}, sigma=0):

    ### Branch: True/False, return 1/0
    if isinstance(j, bool):
        return tc.tensor(int(j)).float(), sigma

    ### Branch: If int/float, return tensor version
    elif isinstance(j, int) or isinstance(j, float):
        return tc.tensor(j).float(), sigma

    ### If observe, update weight sigma
    elif j[0] == "observe":
        d = evaluate(j[1], l = l, rho = rho, sigma = sigma)[0]
        v = evaluate(j[2], l=l, rho = rho, sigma = sigma)[0]
        logp = d.log_prob(v)
        sigma["logW"]+= tc.tensor(logp)
        return v, sigma

    ### If sample, or sample*, evaluate distribution and sample
    elif j[0] == "sample" or j[0] == "sample*":
        val, sig = evaluate(j[1], l = l, rho=rho, sigma = sigma)
        return val.sample(), sig

    ### If: lazy evaluation
    elif j[0] == "if":
        val, sig = evaluate(j[1], l = l , rho= rho, sigma= sigma)
        if(val):
            return evaluate(j[2], l = l, rho=rho, sigma= sigma)
        else:
            return evaluate(j[3], l = l, rho=rho, sigma= sigma)

    ### Let: add to local variable l
    elif j[0] == "let":
        val, sig = evaluate(j[1][1], l, rho, sigma= sigma)
        l[j[1][0]] = val
        return evaluate(j[2], l, rho, sigma= sig)

    ### If it is a string
    elif isinstance(j, str):
        ### If string found in environment, evaluate it
        if j in rho:
            return evaluate(rho[j], l=l, rho=rho, sigma= sigma)
        ### If not, then it is a local variable
        else:
            return l[j], sigma

    ### Lastly, if it is a list
    else:
        opt = rho[j[0]]

        values = []
        for i in range(1, len(j)):
            values.append(evaluate(j[i], l, rho, sigma= sigma)[0])


        ### If rho is a list, not operator, then it's a function dictionary
        if isinstance(opt, list):
            fvars = rho[j[0]][1]
            fexpr = rho[j[0]][0]
            localenv = dict(zip(fvars, values))
            return evaluate(fexpr, l = localenv, rho = rho, sigma= sigma)

        result = opt(*values), sigma

        return result



def evaluate_program(ast, verbose=False):

    env = add_functions(ast.functions)
    env = {**env, **primitives}
    result, sigma = evaluate(ast.program, rho = env, sigma={"logW":0})

    return result, sigma, None
