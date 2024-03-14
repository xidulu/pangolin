import pangolin
from pangolin import transforms
from pangolin.interface import (
    vmap, normal_scale, normal, RV, VMapDist,
    makerv, viz_upstream, exponential, bernoulli, print_upstream,
    Constant, plate, exp, Index, multi_normal_cov, Sum, bernoulli_logit
)
from pangolin import Calculate
from pangolin import dag
import numpy as np
from contrib.auto_vmap import Replace, Merge, Find, AutoVmap, AggregateParents, MergeIndexRVs
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import jax
jax.config.update("jax_enable_x64", True)
from utils import check_same_joint_cond_dist
from copy import deepcopy

def _test_raise_exception(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except AssertionError:
        pass
    else:
        assert False, 'no exception catched'


def case1():
    x = normal(1,1)
    y = [normal(x + i,1) for i in range(3)]
    z = [[normal(yi + i,1) for i in range(4)] for yi in y]
    given_vars = z
    given_vals = [[1.0] * 4 for _ in range(len(y))]
    return z, None, None

def case2():
    w1 = normal_scale(1, 1)
    w2 = normal_scale(2, 1)
    sigma = exp(normal_scale(0, 1))
    x = [normal_scale(w1 * 2.0 + w2 * 3.0, sigma) for _ in range(4)]
    return x, None, None

def case3():
    x = normal(1,1)
    y = []
    z = []
    u = []
    for i in range(2):
        yi = normal(x + i, 1)
        y.append(yi)
        for j in range(3):
            zi = normal(yi + j, 1)
            z.append(zi)
            for k in range(4):
                ui = normal(zi + k, 1)
                u.append(ui)
    return u, None, None

def case4():
    x = [normal(i + 1,1) for i in range(3)]
    y = [normal(x[0],1), normal(x[1],1), normal(x[2],1)]
    z = [normal(x[2],1), normal(x[0],1), normal(x[1],1)]
    return z, None, None

def case5():
    x = normal_scale(1, 1)
    y = {}
    z = {}
    for i in range(3):
        for j in range(3):
            y[(i, j)] = normal_scale(x, i + j)
            z[(i, j)] = normal_scale(x, 1)
            
    O = {}
    for i in range(3):
        for j in range(3):
            O[(i, j)] = bernoulli_logit(y[(i, j)] * z[i, j])
    
    return O, None, None

def case6():
    B = 2 # Batch size
    T = 3 # sequence length
    Q = [multi_normal_cov(np.zeros(T) + b, np.eye(T) / 4) for b in range(B)] # Queries
    K = [multi_normal_cov(np.zeros(T) + b, np.eye(T) / 4) for b in range(B)] # Keys
    V = [vmap(exp)(multi_normal_cov(np.zeros(T) + 1, np.eye(T) / 4)) for b in range(B)] # Values
    outputs = []
    for b in range(B): # b: Batch ID
        output = []
        for t in range(T):
            q = Q[b][t] # The t_th token for query of the b_th sample.
            k = K[b] # Keys of the b_th sample
            similarity_score = q * k # q * k denotes the similarity score
            output.append(
                Sum(0)(similarity_score * V[b]) # sum(score_t * value)
            )
        outputs.append(output)
    return outputs, None, None

def case7():
    D = 2
    x = [normal_scale(0, 1) for _ in range(D)]
    h1 = [normal_scale(sum(x[:i] + x[i + 1:]), 1) for i in range(D)]
    h2 = [normal_scale(sum(h1[:i] + h1[i + 1:]), 1) for i in range(D)]
    return h2, None, None

def case8():
    v = [exp(normal(i, i)) for i in range(2)]
    x = normal(1, 1)
    y = [normal(x + i, 1) for i in range(2)]
    z = [[normal(yi, vj) for yi in y] for vj in v]
    return z, None, None


'''
Cases with observed values
'''

def case9():
    x = normal(1,1)
    y = [normal(x + i,1) for i in range(3)]
    z = [[normal(yi + i,1) for i in range(4)] for yi in y]
    given_vars = z
    given_vals = [[1.0] * 4 for _ in range(len(y))]
    return x, given_vars, given_vals

def case10():
    x = normal(1,1)
    y = [normal(x + i,1) for i in range(3)]
    z = [[normal(yi + i,1) for i in range(4)] for yi in y]
    given_vars = z[0] # Partially observed
    given_vals = [1.0 for _ in range(len(given_vars))]
    return x, given_vars, given_vals

def case11():
    v = [exp(normal(i, i + 1)) for i in range(2)]
    x = normal(1, 1)
    y = [normal(x + i, 1) for i in range(2)]
    z = [[normal(yi, vj) for yi in y] for vj in v]
    given_vars = z
    given_vals = [[1.0 + i] * 2 for i in range(2)]
    return y, given_vars, given_vals


def case12():
    x = normal(1,1)
    y = []
    z = []
    u = []
    for i in range(2):
        yi = normal(x + i, 1)
        y.append(yi)
        for j in range(3):
            zi = normal(yi + j, 1)
            z.append(zi)
            for k in range(4):
                ui = normal(zi + k, k + 1)
                u.append(ui)
    given_vars = u[1:3] + u[5:6] + u[-2:]
    given_vals = [i for i in range(5)]
    return y + [x], given_vars, given_vals


def case13():
    x = normal(1,1)
    y = []
    z = []
    u = []
    for i in range(2):
        yi = normal(x + i, 1)
        y.append(yi)
        for j in range(3):
            zi = normal(yi + j, 1)
            z.append(zi)
            for k in range(4):
                ui = normal(zi + k, k + 1)
                u.append(ui)
    given_vars = u[:2] + u[4:6] + u[-2:]
    given_vals = [i for i in range(6)]
    return y, given_vars, given_vals

def case14():
    x = normal(1,1)
    y = []
    z = []
    u = []
    for i in range(2):
        yi = normal(x + i, 1)
        y.append(yi)
        for j in range(3):
            zi = normal(yi + j, 1)
            z.append(zi)
            for k in range(4):
                ui = normal(zi + k, k + 1)
                u.append(ui)
    given_vars = [u[0], u[3], u[5]]
    given_vals = [i for i in range(3)]
    return y, given_vars, given_vals


def test_AutoVmap_base():
    """
    Prior check

    Idea: Check following things for the output of AutoVmap
    1. If find(transformed_vars) == 0, i.e. no more vmappable groups
    2. If transformed_vars has the same length as vars
    3. If the empirical mean and cov (from ancestral sampling) matches with the original model.
    """
    def _test_model(vars, given_vars=[], given_vals=[], order=0):
        vars, tree_def = tree_flatten(vars)
        given_vars = tree_flatten(given_vars)[0]
        given_vals = tree_flatten(given_vals)[0]
        transformed_var, transformed_given_var, transformed_given_vals = AutoVmap(
            vars, given_vars, given_vals, order
        )
        assert len(transformed_var) == len(vars)
        assert len(Find(transformed_var)) == 0
        # assert all(isinstance(var.cond_dist, Index) for var in transformed_var)
        check_same_joint_cond_dist(vars, transformed_var, verbose=True)

    cases = [case1, case2, case3, case4, case5, case6, case7, case8]
    for case in cases:
        _test_model(case()[0], order=0)
        _test_model(case()[0], order=-1)
    
    print('AutoVmap basic test passed')


def test_AutoVmap_posterior():
    """
    Posterior check

    Check the base conditions, additionally
    check whether the conditional distribuiton preserves when observed values are provided
    """
    def _test_model(case, order=-1):
        vars, given_vars, given_vals = map(lambda v: tree_flatten(v)[0], case())
        transformed_var, transformed_given_var, transformed_given_vals = AutoVmap(
            *map(lambda v: tree_flatten(v)[0], case()), order
        )
        assert len(transformed_var) == len(vars)
        assert len(Find(transformed_var)) == 0
        assert all(not isinstance(var.cond_dist, Index) for var in transformed_given_var) # No indexing node!!!
        assert len(transformed_given_var) <= len(given_vars)
        # Check same posterior
        check_same_joint_cond_dist(vars, transformed_var,
                                   given_vars, given_vals,
                                   transformed_given_var, transformed_given_vals,
                                   verbose=True)

    cases = [case10, case11, case12, case13, case14]
    for case in cases:
        _test_model(case, 0)
        _test_model(case, -1)
    
    print('AutoVmap posterior test passed')


test_AutoVmap_base()
test_AutoVmap_posterior()