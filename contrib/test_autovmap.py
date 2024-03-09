import pangolin
from pangolin import transforms
from pangolin.interface import (
    vmap, normal_scale, normal, RV, VMapDist,
    makerv, viz_upstream, exponential, bernoulli, print_upstream,
    Constant, plate, exp
)
from pangolin import Calculate
from pangolin import dag
import numpy as np
from contrib.auto_vmap_general import Replace, Merge, Find, AutoVmap, AggregateParents, MergeIndexRVs
from jax.tree_util import tree_flatten, tree_unflatten

from utils import check_same_joint_cond_dist

def _test_raise_exception(func, *args, **kwargs):
    try:
        func(*args, **kwargs)
    except AssertionError:
        pass
    else:
        assert False, 'no exception catched'


def test_find():
    x = normal(0,1)
    y = [normal(x,1) for i in range(5)]
    z = [[normal(yi,1) for i in range(6)] for yi in y]
    all_vars = tree_flatten([x, y, z])[0]
    grouped_rvs = Find(all_vars)
    assert len(grouped_rvs) == 6

    # z now split into two groups: Observed and not observed
    grouped_rvs = Find(all_vars, given_vars=[z[0][0], z[0][1], z[0][2]])
    assert len(grouped_rvs) == 7

    # Test vector case
    x = vmap(normal, in_axes=None, axis_size=3)(0,1)
    y = [vmap(lambda loc: normal_scale(loc, 1))(x) for _ in range(5)]
    grouped_rvs = Find(y)
    assert len(grouped_rvs) == 1

    # Test "parent-is-index-node" cases
    x = vmap(normal, in_axes=None, axis_size=3)(0,1)
    y = [vmap(lambda loc: normal_scale(loc, 1))(x) for _ in range(5)]
    z = [[vmap(lambda loc: normal_scale(loc, 1))(y[i]) for _ in range(6)] for i in range(5)]
    grouped_rvs = Find(tree_flatten([x, y, z])[0])
    assert len(grouped_rvs) == 6 # 5 groups from z, and 1 group from y

    # Test deterministic node cases
    x = normal(0,1)
    y = [normal(1, exp(x)) for i in range(5)]
    grouped_rvs = Find(tree_flatten([x, y])[0])
    # The exp nodes are not in tree_flatten([x, y]), therefore no vmappable groups
    assert len(grouped_rvs) == 0 

    x = exp(normal(0,1))
    y = [normal(1, x) for i in range(5)]
    grouped_rvs = Find(tree_flatten([x, y])[0])
    assert len(grouped_rvs) == 1 # All y belongs to the same group

    # Test index node case
    x = vmap(normal, in_axes=None, axis_size=10)(0,1)
    y = [x[i] for i in range(3)]
    grouped_rvs = Find(y)
    assert len(grouped_rvs) == 0 # All y belongs to the same group

    # Some other random cases
    x = [normal(i,1) for i in range(3)]
    y = [normal(x[0],1), normal(x[1],1), normal(x[2],1)]
    z = [normal(x[2],1), normal(x[0],1), normal(x[1],1)]
    grouped_rvs = Find(tree_flatten([x, y, z])[0])
    assert len(grouped_rvs) == 4 # x, (y[0], z[1]), (y[1], z[2]), (y[2], x[0])


    print('Find test passed')


def test_merge_indexed_rv():
    y = plate(N=7)(lambda:
            plate(N=5)(lambda:
               plate(N=6)(lambda:
                          normal_scale(0,1))))
    indexed_nodes = [y[:, 2, :], y[:, 3, :], y[:, 1, :], y[:, 2, :]]
    merged_rv, indexed_dim = MergeIndexRVs(indexed_nodes)
    assert merged_rv.shape == (7, 4, 6)
    assert indexed_dim == 1
    assert all(merged_rv.parents[1].cond_dist.value == np.array([2, 3, 1, 2]))

    indexed_nodes = [y[:, :, 1], y[:, :, 0], y[:, :, 1], y[:, :, 0]]
    merged_rv, indexed_dim = MergeIndexRVs(indexed_nodes)
    assert merged_rv.shape == (7, 5, 4)
    assert indexed_dim == 2
    assert all(merged_rv.parents[1].cond_dist.value == np.array([1, 0, 1, 0]))

    indexed_nodes = [y[:, :, i] for i in range(6)]
    merged_rv, indexed_dim = MergeIndexRVs(indexed_nodes)
    assert merged_rv.shape == (7, 5, 6)
    assert indexed_dim == 2
    assert (merged_rv == y)

    print('Merge indexed node good cases passed')

    indexed_nodes = [y[:, :, 0], y[1, :, :], y[:, 0, :]]
    _test_raise_exception(MergeIndexRVs, indexed_nodes)

    print('Merge indexed node bad cases passed')


def test_merge():
    x = normal_scale(1, 1)
    y = [normal_scale(x, i) for i in range(10)]
    vmapped_y, _ = Merge(y)
    assert vmapped_y.cond_dist.axis_size == 10
    vmapped_y, stacked_Y = Merge(y, [np.array(1)] * 10)
    assert vmapped_y.cond_dist.axis_size == 10
    assert stacked_Y.shape == (10,)

    z = [normal_scale(vmapped_y[i], i) for i in [1, 3, 5, 7]]
    vmapped_z, _ = Merge(z)
    assert vmapped_z.cond_dist.axis_size == 4
    check_same_joint_cond_dist(z, [vmapped_z[i] for i in range(4)])
    print('Merge good cases passed')
    
    _test_raise_exception(Merge, [normal_scale(0, 1), exponential(1)])

    x = normal_scale(0, 1)
    y = normal_scale(0, 1)
    _test_raise_exception(Merge, [normal_scale(x, 1), normal_scale(y, 1)])

    x = makerv([1, 2, 3])
    y = makerv([[1]])
    _test_raise_exception(Merge, [x, y])
    print('Merge bad cases passed')



'''
Begin testing end-to-end AutoVmap
'''



def test_AutoVmap():
    """
    Idea: Check following things for the output of AutoVmap
    1. If find(transformed_vars) == 0, i.e. no more vmappable groups
    2. If all transformed_vars are indexed from a vmapRV
    3. If transformed_vars has the same length as vars
    3. If the empirical mean and cov (from ancestral sampling) matches with the original model.
    """
    pass

# def test_AutoVmap_case1():
#     # Check same joint distribution with no observed values
#     x = normal(1, 1)
#     y = [normal(x, 1) for i in range(5)]
#     z = [[normal(yi, 1) for i in range(6)] for yi in y]
#     all_vars, tree_def = tree_flatten([x, y, z])
#     transformed_var, transformed_given_var, transformed_given_vals = AutoVmap(all_vars, [], [])
#     check_same_joint_cond_dist(all_vars, transformed_var)
#     new_var = tree_unflatten(tree_def, transformed_var)
#     old_var = tree_unflatten(tree_def, all_vars)
#     check_same_joint_cond_dist(old_var, new_var)
#     print('AutoVmap perserves joint distribution for unobserved case passed')
    

test_find()
test_merge_indexed_rv()
test_merge()
test_AutoVmap()