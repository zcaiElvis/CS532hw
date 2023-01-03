import torch as tc

def vector(*x):
    # NOTE: This must support both lists and vectors
    try:
        result = tc.stack(x)
    except:
        result = list(x)
    return result

def hashmap(*x):
    _keys = [key for key in x[0::2]]
    keys = []
    for key in _keys: # Torch tensors cannot be dictionary keys, so convert here
        if type(key) is tc.Tensor: key = float(key)
        keys.append(key)
    values = [value for value in x[1::2]]
    return dict(zip(keys, values))

# Primative function dictionary


def put(vec,pos,val):
    if tc.is_tensor(pos):
        vec[int(pos.item())] = val
    else:
        vec[pos] = val
    return(vec)

def get(a,b):
    if(tc.is_tensor(b)):
        return a[int(b.item())]
    else:
        return a[b]

def append(a,b):
    return(tc.cat((a, tc.tensor([b]))))

class dirac:
    def __init__(self, val):
        self.val = val

    def log_prob(self, v):
        if tc.isclose(v, self.val, atol=1e-01):
            return tc.tensor(0)
        else:
            return tc.tensor(float('-inf'))



# NOTE: You should complete this
primitives = {

    # Comparisons
    '<': tc.lt,
    '<=': tc.le,
    '=':tc.equal,
    'and': lambda a, b: tc.logical_and(tc.tensor(a), tc.tensor(b)),
    'or': lambda a, b: tc.logical_or(tc.tensor(a), tc.tensor(b)),
    'abs': tc.abs,
    # ...

    # Math
    '+': tc.add,
    '*': tc.mul,
    'sqrt': tc.sqrt,
    '/': tc.divide,


    # List operation
    'get': get,
    'put': put,
    'first': lambda a: a[0],
    'second': lambda a: a[1],
    'rest': lambda a: a[1:],
    'last': lambda a: a[len(a)-1],
    'append': append,
    # ...

    # Containers
    'vector': vector,
    'hash-map': hashmap,
    # ...

    # Matrices
    'mat-mul': tc.matmul,
    'mat-transpose': tc.t,
    'mat-tanh': tc.tanh,
    'mat-add': tc.add,
    'mat-repmat': lambda mat, dim1, dim2: mat.repeat(int(dim1), int(dim2)),
    # ...

    # Distributions
    'normal': tc.distributions.Normal,
    'beta': tc.distributions.Beta,
    'exponential': tc.distributions.Exponential,
    'uniform-continuous':tc.distributions.Uniform,
    'discrete':tc.distributions.Categorical,
    'gamma': tc.distributions.Gamma,
    'dirichlet': tc.distributions.Dirichlet,
    'flip':tc.distributions.Bernoulli,
    'dirac': lambda x: dirac(x)
    # ...

}
