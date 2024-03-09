import pangolin
from pangolin import transforms
from pangolin.interface import (
    vmap, normal_scale, normal, RV, VMapDist,
    makerv, viz_upstream, exponential, bernoulli, print_upstream,
    Constant
)
from pangolin import Calculate
from pangolin import dag
import numpy as np
from contrib.auto_vmap_inaxes_None import Replace, Merge, Find, AutoVmap
from jax.tree_util import tree_flatten, tree_unflatten

from utils import check_same_joint_cond_dist

def _all_the_same(x):
    return all (_x == x[0] for _x in x)

def check_same_joint_cond_var(
        old_var, old_given_var, old_given_vals,
        new_var, new_given_var, new_given_vals    
    ):
    '''
    The function would:
    1. Generate samples from unobserved old_var/new_var, denoted as x_old and x_new
    2. Compute and compare p(old_var=x_old | old_given_var=old_given_vals) and
        p(new_var=x_new | new_given_var=new_given_vals)
    '''
    calc = Calculate("jags",niter=10000)
    samples_old_dag = calc.sample(old_var, old_given_var, old_given_vals)
    samples_new_dag = calc.sample(new_var, new_given_var, new_given_vals)
    

def test_find():
    x = normal(0,1)
    y = [normal(x,1) for i in range(5)]
    z = [[normal(yi,1) for i in range(6)] for yi in y]
    all_vars = tree_flatten([x, y, z])[0]
    grouped_rvs = Find(all_vars)
    assert len(grouped_rvs) == 6

    grouped_rvs = Find(all_vars, given_vars=[z[0][0], z[0][1], z[0][2]])
    assert len(grouped_rvs) == 7

    # Test vector case
    x = vmap(normal, in_axes=None, axis_size=3)(0,1)
    y = [vmap(lambda loc: normal_scale(loc, 1))(x) for _ in range(5)]
    grouped_rvs = Find(y)
    assert len(grouped_rvs) == 1

    print('Find test passed')

def test_replace():
    x = normal_scale(0, 1)
    y = normal_scale(x, 1)
    z = normal_scale(y, 1)
    new_x, new_y, new_z = Replace([x, y, z], [y], [normal_scale(x, 2)])
    assert new_y.parents[0] == new_x
    assert new_z.parents[0] == new_y
    assert new_y.parents[1].cond_dist == Constant(2)

    x = normal_scale(0, 1)
    y = normal_scale(x, 1)
    z = normal_scale(y, 1)
    new_x, new_y, new_z = Replace([x, y, z], [x], [normal_scale(1, 1)])
    assert new_y.parents[0] == new_x
    assert new_z.parents[0] == new_y
    assert new_x.parents[0].cond_dist == Constant(1)

    x = normal_scale(0, 1)
    y = normal_scale(x, 1)
    z = [normal_scale(y, 1) for _ in range(10)]
    new_x, new_y, *new_zs = Replace([x, y] + z, [x], [normal_scale(1, 1)])
    assert new_y.parents[0] == new_x
    for new_z in new_zs:
        assert new_z.parents[0] == new_y
    assert new_x.parents[0].cond_dist == Constant(1)

    new_x, new_y, *new_zs = Replace([x, y] + z, [y], [normal_scale(x, 2)])
    assert new_y.parents[0] == new_x
    for new_z in new_zs:
        assert new_z.parents[0] == new_y
    assert new_y.parents[1].cond_dist == Constant(2)
    print('Replace test passed')

def test_merge_rvs():
    x = normal_scale(0, 1)
    y = [normal_scale(x, 1) for _ in range(10)]
    vmapped_y, _ = Merge(y)
    assert vmapped_y.cond_dist.axis_size == 10
    vmapped_y, stacked_Y = Merge(y, [np.array(1)] * 10)
    assert vmapped_y.cond_dist.axis_size == 10
    assert stacked_Y.shape == (10,)
    print('Merge test passed')


def test_AutoVmap_case0():
    # Check AutoVmap actually vmapped things together
    z = normal(0, 1)
    x = [normal_scale(z, 1) for _ in range(2)]
    y1 = [normal_scale(x[0], 2) for _ in range(3)]
    y2 = [normal_scale(x[1], 3) for _ in range(3)]
    given_vals = [np.array(i * 1.0) - 5 for i in range(3)]
    all_vars, tree_def = tree_flatten([z, x, y1, y2])
    given_vars, _ = tree_flatten(y1)
    transformed_var, transformed_given_var, transformed_given_vals = AutoVmap(
        all_vars, given_vars, given_vals
    )
    # One elements in transformed given_var and given_val should be of shape (3,)
    assert len(transformed_given_var) == 1
    assert len(transformed_given_vals) == 1
    assert transformed_given_vals[0].shape == (3,)

    # Transformed var and input vars having the same length
    assert len(all_vars) == len(transformed_var)

    # Assert there are three VMapRVs in transformed DAG
    all_nodes = dag.upstream_nodes(transformed_var)
    num_vmaprv = sum([isinstance(node.cond_dist, VMapDist) for node in all_nodes])
    print(num_vmaprv)
    assert num_vmaprv == 3 # x, y1, y2

    print('AutoVmap structure test passed')


def test_AutoVmap_case1():
    # Check same joint distribution with no observed values
    x = normal(1, 1)
    y = [normal(x + i, 1) for i in range(5)]
    z = [[normal(yi + i, 1) for i in range(6)] for yi in y]
    all_vars, tree_def = tree_flatten([x, y, z])
    transformed_var, transformed_given_var, transformed_given_vals = AutoVmap(all_vars, [], [])
    check_same_joint_cond_dist(all_vars, transformed_var)
    new_var = tree_unflatten(tree_def, transformed_var)
    old_var = tree_unflatten(tree_def, all_vars)
    check_same_joint_cond_dist(old_var, new_var)
    print('AutoVmap perserves joint distribution for unobserved case passed')


def test_AutoVmap_case2():
    # Check same joint distributio with no observed values
    # Contain observed values
    z = normal(0, 1)
    x = [normal_scale(z, 1) for _ in range(2)]
    y1 = [normal_scale(x[0], 2) for _ in range(3)]
    y2 = [normal_scale(x[1], 3) for _ in range(3)]
    given_vals = [np.array(i * 1.0) + 5 for i in range(3)]
    all_vars, tree_def_allvars = tree_flatten([z, x, y1, y2])
    given_vars, tree_def_gvars = tree_flatten(y2)
    transformed_var, transformed_given_var, transformed_given_vals = AutoVmap(
        all_vars, given_vars, given_vals
    )
    check_same_joint_cond_dist(
        all_vars, transformed_var,
        y2, given_vals,
        transformed_given_var, transformed_given_vals # Both are lists of only one element
    )
    print('AutoVmap perserves joint distribution for observed case passed')


test_find()
test_merge_rvs()
test_replace()
test_AutoVmap_case0()
test_AutoVmap_case1()
test_AutoVmap_case2()