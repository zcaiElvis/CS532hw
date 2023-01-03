# Standard imports
import torch as tc
import json

# Project imports
from primitives import primitives


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




def evaluate_vi(j, l={}, rho={}, sigma={}):

    ### Branch: True/False, return 1/0
    if isinstance(j, bool):
        t = tc.tensor(j).float()
        return t, sigma

    ### Branch: If int/float, return tensor version
    elif isinstance(j, int) or isinstance(j, float):
        return tc.tensor(j).float(), sigma

    elif tc.is_tensor(j):
        return j, sigma

    ### If observe, update weight sigma
    elif j[0] == "observe" or j[0] == "observe*":
        d = evaluate_vi(j[1], l = l, rho = rho, sigma = sigma)
        v = evaluate_vi(j[2], l=l, rho = rho, sigma = sigma)
        logp = d.log_prob(v)
        sigma["logW"]+= logp
        return logp, sigma ### supposed to return v, not logp

    ### If sample, or sample*, evaluate_vi distribution and sample
    elif j[0] == "sample" or j[0] == "sample*":
        val, sig = evaluate_vi(j[1], l = l, rho=rho, sigma = sigma)
        return val.sample(), sig

    ### If: lazy evaluation
    elif j[0] == "if":
        val, sig = evaluate_vi(j[1], l = l , rho= rho, sigma= sigma)
        if(val):
            return evaluate_vi(j[2], l = l, rho=rho, sigma= sigma)
        else:
            return evaluate_vi(j[3], l = l, rho=rho, sigma= sigma)

    ### Let: add to local variable l
    elif j[0] == "let":
        val, sig = evaluate_vi(j[1][1], l, rho, sigma= sigma)
        l[j[1][0]] = val
        return evaluate_vi(j[2], l, rho, sigma= sig)

    ### If it is a string
    elif isinstance(j, str):
        ### If string found in environment, evaluate_vi it
        if j in rho:
            return evaluate_vi(rho[j], l=l, rho=rho, sigma= sigma)
        ### If not, then it is a local variable
        else:
            return l[j], sigma

    ### Lastly, if it is a list
    else:
        opt = rho[j[0]]

        values = []
        for i in range(1, len(j)):
            val, sigma = evaluate_vi(j[i], l, rho, sigma= sigma)
            values.append(val)
            # values.append(evaluate_vi(j[i], l, rho, sigma= sigma)[0])


        ### If rho is a list, not operator, then it's a function dictionary
        if isinstance(opt, list):
            fvars = rho[j[0]][1]
            fexpr = rho[j[0]][0]
            localenv = dict(zip(fvars, values))
            return evaluate_vi(fexpr, l = localenv, rho = rho, sigma= sigma)

        result = opt(*values), sigma

        return result



def evaluate_program(ast, verbose=False):

    env = add_functions(ast.functions)
    env = {**env, **primitives}
    result, sigma = evaluate_vi(ast.program, rho = env, sigma={"logW":0})

    return result, sigma, None
