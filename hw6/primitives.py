# Standard imports
import torch as tc
from pyrsistent._pmap import PMap
from pyrsistent._plist import PList, _EmptyPList as EmptyPList
from pyrsistent import pmap, plist

def vector(*x):
    # This needs to support both lists and vectors
    try:
        result = tc.stack(x) # NOTE: Important to use stack rather than tc.tensor
    except: # NOTE: This except is horrible, but necessary for list/vector ambiguity
        result = plist(x)
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
    previous_address, current_addreess, continuation = x[0], x[1], x[2]
    return continuation(previous_address+'-'+current_addreess)

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


# Primitive function dictionary
# NOTE: Fill this in
primitives = {

    # HOPPL
    'push-address': push_address,

    # Comparisons
    '<': lambda *x: x[-1](tc.lt(*x[1:-1])),
    '>': lambda *x: x[-1](tc.gt(*x[1:-1])),

    # Maths
    '+': lambda *x: x[-1](tc.add(*x[1:-1])),
    'sqrt': lambda *x: x[-1](tc.sqrt(*x[1:-1])),
    '*': lambda *x: x[-1](tc.multiply(*x[1:-1])),
    '/': lambda *x: x[-1](tc.divide(*x[1:-1])),
    '-': lambda *x: x[-1](tc.subtract(*x[1:-1])),
    'log': lambda *x: x[-1](tc.log(*x[1:-1])),

    # Containers
    'vector': lambda *x: x[-1](vector(*x[1:-1])),
    'hash-map': lambda *x: x[-1](hashmap(*x[1:-1])),
    'get': lambda *x: x[-1](get(*x[1:-1])),
    'put': lambda *x: x[-1](put(*x[1:-1])),
    'first': lambda *x : x[-1](x[1][0]),
    'last': lambda *x: x[-1](x[1][-1]),
    'append': lambda *x: x[-1](append(*x[1:-1])),
    'empty?': lambda *x: x[-1](isempty(*x[1:-1])),
    'rest': lambda *x: x[-1](x[1][1:]),
    'conj': lambda *x: x[-1](conj(*x[1:-1])),
    'peek': lambda *x: x[-1](x[1][0]),

    # Matrices
    'mat-transpose': lambda *x: x[-1](tc.transpose(*x[1:-1], 0, 1)),

    # Distributions
    'normal': lambda *x: x[-1](tc.distributions.Normal(*x[1:-1])),
    'beta': lambda *x: x[-1](tc.distributions.Beta(*x[1:-1])),
    'exponential': lambda *x: x[-1](tc.distributions.Exponential(*x[1:-1])),
    'uniform-continuous': lambda *x: x[-1](tc.distributions.Uniform(*x[1:-1])),
    'flip': lambda *x: x[-1](tc.distributions.Bernoulli(*x[1:-1])),
    'discrete': lambda *x: x[-1](tc.distributions.Categorical(*x[1:-1])),
}