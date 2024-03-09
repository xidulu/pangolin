import pangolin
from pangolin import transforms
from pangolin.interface import (
    vmap, normal_scale, normal, RV, VMapDist,
    makerv, viz_upstream, exponential, bernoulli, print_upstream,
    Constant, exp, abs
)
from pangolin import Calculate
from pangolin import dag
import numpy as np
from univaraite_collapse import FindUnivariateNode, CollapseTransform
from jax.tree_util import tree_flatten, tree_unflatten

from utils import check_same_joint_cond_dist

def test_find_node():
    z = normal_scale(0, 1)
    y = [normal_scale(0, exp(z)) for _ in range(5)]
    output = FindUnivariateNode(y, exp)[0]
    assert len(output) == len(y)

def test_transform_0():
    # Case 0.0
    x = normal(1, 1)
    y = [normal(x, i + 1) for i in range(2)]
    # z contains duplicated exp in the parents
    z = [[normal(exp(yi), 1) for i in range(3)] for yi in y]
    all_vars, tree_def = tree_flatten([x, y, z])
    transformed_vars, _, _ = CollapseTransform(all_vars, [], [])
    # Check if the exp nodes are removed through inspecting the z's parents,
    x_t, y_t, z_t = tree_unflatten(tree_def, transformed_vars)
    for branch in z_t:
        # Assert all branch nodes share the same exp parents
        assert all([leaf.parents[0] == branch[0].parents[0] for leaf in branch])

    # Case 0.1
    x = normal(1, 1)
    y = [normal(i, exp(exp(exp(x)))) for i in range(3)] # 9 exp nodes in total
    all_vars, tree_def = tree_flatten([x, y])
    transformed_vars, _, _ = CollapseTransform(all_vars, [], []) 
    # After collapsing, there should only be 3 exp nodes left
    assert sum([rv.cond_dist == exp for rv in dag.upstream_nodes(transformed_vars)]) == 3

    print('Check additional exp nodes being eliminated')

def test_transform_1():
    x = normal(1, 1)
    y = [normal(x, i + 1) for i in range(2)]
    # z contains duplicated exp in the parents
    z = [[normal(exp(yi), 1) for i in range(3)] for yi in y]
    all_vars, tree_def = tree_flatten([x, y, z])
    transformed_vars, _, _ = CollapseTransform(all_vars, [], [])
    # Check if joint distribution is preseved
    check_same_joint_cond_dist(all_vars, transformed_vars, verbose=False) 
    print('Same joint distribution checked (no observed variables)')

def test_transform_2():
    x = normal(1, 1)
    y = [normal(x, i + 1) for i in range(2)]
    # z contains duplicated exp in the parents
    z = [[normal(exp(yi), 1) for i in range(3)] for yi in y]
    given_var = tree_flatten(z)[0]
    all_vars = tree_flatten([x, y, z])[0]
    given_vals = [np.array(i) - 3.0  for i in range(6)]
    transformed_vars, transformed_gvars, transformed_gvals = CollapseTransform(
        all_vars, given_var, given_vals
    )
    # Check if joint distribution is preseved
    check_same_joint_cond_dist(all_vars, transformed_vars,
                               given_var, given_vals,
                               transformed_gvars, transformed_gvals,
                               verbose=False) 
    print('Same joint distribution checked (with observed variables)')




test_find_node()
test_transform_0()
test_transform_1()
test_transform_2()