# Standard imports
import torch as tc
from pyrsistent import pmap
from time import time
import sys

# Project imports
from primitives import primitives

class Env():
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.data = pmap(zip(parms, args))
        self.outer = outer
        if outer is None:
            self.level = 0
        else:
            self.level = outer.level+1

    def __getitem__(self, item):
        return self.data[item]

    def find(self, var):
        "Find the innermost Env where var appears."
        if (var in self.data):
            return self
        else:
            if self.outer is not None:
                return self.outer.find(var)
            else:
                raise RuntimeError('var "{}" not found in outermost scope'.format(var))

    def print_env(self, print_lowest=False):
        print_limit = 1 if print_lowest == False else 0
        outer = self
        while outer is not None:
            if outer.level >= print_limit:
                print('Scope on level ', outer.level)
                if 'f' in outer:
                    print('Found f, ')
                    print(outer['f'].body)
                    print(outer['f'].parms)
                    print(outer['f'].env)
                print(outer,'\n')
            outer = outer.outer



class Procedure(object):
    'A user-defined HOPPL procedure'
    def __init__(self, params:list, body:list, sig:dict, env:Env):
        self.params, self.body, self.sig, self.env = params, body, sig, env
    def __call__(self, *args):
        return eval(self.body, self.sig, Env(self.params, args, self.env))



def standard_env():
    "An environment with some Scheme standard procedures."
    env = Env(primitives.keys(), primitives.values())
    return env

def eval(e, sig:dict, env:Env, verbose=False):


    if isinstance(e, bool):
        t = tc.tensor(e).float()
        return t
    
    elif isinstance(e, int) or isinstance(e, float):
        return tc.tensor(e).float()

    elif tc.is_tensor(e):
        return e

    elif isinstance(e, str):
        try:
            ### If already defined, return corresponding value
            return env.find(e)[e]

        except:
            # sig['address'] = sig['address'] + "-" + e
            addr = sig['address'] + "-" + e
            sig.set('address', addr)
            ### If not, evaluate it below
            return e

    op, *args = e

    if op == "if":
        (test, conseq, alt) = args
        if eval(test, sig, env):
            exp = conseq
        else:
            exp = alt

        return eval(exp, sig, env)

    elif op == "sample" or op == "sample*":
        d = eval(args[1], sig, env)
        s = d.sample()
        k = eval(args[2], sig, env)
        addr = eval(args[0], sig, env)

        sig = pmap({
            "type" : "sample",
            "address" : addr,
            "logW" : sig["logW"]
        })

        k.sig=sig

        return k, [s], sig

    elif op == "observe" or op == "observe*":
        d = eval(args[1], sig, env)
        v = eval(args[2], sig, env)
        k = eval(args[3], sig, env)
        logp = d.log_prob(v)
        addr = eval(args[0], sig, env)

        sig = pmap({
            "type" : "observe",
            "address" : addr,
            "logW" : sig['logW'] + logp
        })

        k.sig = sig
        
        return k, [v], sig


    elif op == "fn":
        (parms, body) = args
        return Procedure(parms, body, sig, env)

    
    else:
        proc = eval(op, sig, env)
        args = []
        for arg in e[1:]:
            args.append(eval(arg, sig, env))

        if isinstance(proc, str):
            raise Exception("{} is not a procedure".format(proc))

        return proc(*args)


def evaluate(ast:dict, sig = pmap({'logW':tc.tensor(0.), 'type': None, 'address': "start"}), run_name='start', verbose=False):
    '''
    Evaluate a HOPPL program as desugared by daphne
    Args:
        ast: abstract syntax tree
    Returns: The return value of the program
    '''
    env = standard_env()
    output = lambda x: x # Identity function, so that output value is identical to output
    exp = eval(ast, sig, env, verbose)(run_name, output) # NOTE: Must run as function with a continuation
    while type(exp) is tuple: # If there are continuations the exp will be a tuple and a re-evaluation needs to occur
        func, args, sig = exp
        exp = func(*args)
    return exp, sig