# Standard imports
import torch as tc
from pyrsistent import pmap, pvector

def vector(*x):
    # This needs to support both lists and vectors
    try:
        result = tc.stack(x) # NOTE: Important to use stack rather than tc.tensor
    except:
        result = pvector(x)
    return result


def hashmap(*x):
    # This is a dictionary
    keys, values = x[0::2], x[1::2]
    checked_keys = []
    for key in keys: # Torch tensors cannot be dictionary keys, so convert here
        if type(key) is tc.Tensor: key = float(key)
        checked_keys.append(key)
    dictionary = dict(zip(checked_keys, values))
    hashmap = pmap(dictionary)
    return hashmap


def push_address(*x):
    # Concatenate two addresses to produce a new, unique address
    previous_address, current_addreess = x[0], x[1]
    new_address = previous_address+'-'+current_addreess
    return new_address


def get(l,val):
    if(tc.is_tensor(val)):
        return l[int(val.item())]
    else:
        return l[val]

def put(vec,pos,val):

    if isinstance(pos, str):
        if tc.is_tensor(vec):
            vec_new = vec.detach().clone()
            vec_new[pos] = val
            return vec_new
        else:
            vec_new = vec.set(pos,val)
            return vec_new
        
    else:
        pos = int(pos.item())
        if tc.is_tensor(vec):
            vec_new = vec.detach().clone()
            vec_new[pos] = val
            return vec_new
        else:
            vec_new = vec.set(pos,val)
            return vec_new


def append(a,b):
    if not tc.is_tensor(a):
        new_a = tc.tensor(a)
    else:
        new_a = a.detach().clone()

    return tc.cat((new_a, tc.tensor([b]))).clone()

def conj(a,b):
    if not tc.is_tensor(a):
        new_a = tc.tensor(a)
    else:
        new_a = a.detach().clone()

    return tc.cat((tc.tensor([b]), new_a)).clone()


def isempty(vec):
    if tc.is_tensor(vec):
        return tc.numel(vec) == 0
    else:
        return len(vec) == 0

# Primative function dictionary
# NOTE: Fill this in
primitives = {

    # HOPPL
    'push-address' : push_address,

    # Comparisons
    '<': lambda *x: tc.lt(*x[1:]),
    '>': lambda *x: tc.gt(*x[1:]),
    # Maths
    '+': lambda *x: tc.add(*x[1:]),
    'sqrt': lambda *x: tc.sqrt(*x[1:]),
    '*': lambda *x: tc.multiply(*x[1:]),
    '/': lambda *x: tc.divide(*x[1:]),
    '-': lambda *x: tc.subtract(*x[1:]),
    'log': lambda *x: tc.log(*x[1:]),

    # Containers
    'vector': lambda *x: vector(*x[1:]),
    'hash-map': lambda *x: hashmap(*x[1:]),
    'get': lambda *x: get(*x[1:]),
    'put': lambda *x: put(*x[1:]),
    'first': lambda *x: x[1][0],
    'last': lambda *x: x[1][-1],
    'append': lambda *x: append(*x[1:]),
    'empty?': lambda *x: isempty(*x[1:]),
    'rest': lambda *x: x[1][1:],
    'conj': lambda *x: conj(*x[1:]),
    'peek': lambda *x: x[1][0],

    # Matrices
    'mat-transpose': lambda *x: tc.transpose(*x[1:], 0, 1),

    # Distributions
    'normal': lambda *x: tc.distributions.Normal(*x[1:]),
    'beta': lambda *x: tc.distributions.Beta(*x[1:]),
    'exponential': lambda *x: tc.distributions.Exponential(*x[1:]),
    'uniform-continuous': lambda *x: tc.distributions.Uniform(*x[1:]),
    'flip': lambda *x: tc.distributions.Bernoulli(*x[1:]),
    'discrete': lambda *x: tc.distributions.Categorical(*x[1:]),

}