import pangolin
from pangolin import transforms
from pangolin.interface import (
    vmap, normal_scale, normal, RV, VMapDist,
    makerv, viz_upstream, exponential, bernoulli, print_upstream,
    Constant, exp, abs
)
from pangolin import Calculate
from pangolin import dag
import jax
from jax.tree_util import tree_map
from jax import numpy as jnp
# from itertools import groupby
import time
import numpy as np
from collections import defaultdict
from contrib.utils import Replace

def groupby(l, key=lambda x: x):
    d = defaultdict(list)
    for item in l:
        d[key(item)].append(item)
    return d.items()

def FindUnivariateNode(vars, node_type):
    """
    Find all univariate transformation nodes of node_type that share the same parents
    in the DAG referenced by vars.
    """
    all_vars = dag.upstream_nodes(vars)
    univariate_nodes = filter(lambda rv: rv.cond_dist == node_type, all_vars)
    univariate_nodes = groupby(univariate_nodes, lambda rv: rv.parents[0])
    groupped_rvs = []
    for k, v in univariate_nodes:
        if len(v) > 1:
            groupped_rvs.append(v)
    return groupped_rvs


def CollapseTransform(vars, given_vars, given_vals, node_type=exp):
    """
    Inputs:
    vars: A list of RVs that don't contain any RV of node_type
    given_vars: A subset of vars that denotes observed RVs
    given_vals: A list with the same length as given_vars that correspond to the observed values.
    
    Return:
    transformed_vars: RVs with reduncant univariate transformation node in the DAG removed.
    transformed_given_vars: New observed RVs in the transformed DAG, same length as given_vars.
    transformed_given_vals: Same is given_vals
    """
    assert len(given_vars) == len(given_vals)
    assert not any([rv.cond_dist == node_type for rv in vars])
    assert not any([rv.cond_dist == node_type for rv in given_vars])
    assert all([rv in vars for rv in given_vars])
    observed_rv_idx = [vars.index(rv) for rv in given_vars]
    while True:
        # We will handle the "collapse-able" groups one by one, as the DAG could 
        # be modified after calling Replace so we need to run the searching again.
        node_groups = FindUnivariateNode(vars, node_type)
        if len(node_groups) < 1:
            break
        group = node_groups[0] # Work on the first group found
        old_rvs = group
        new_rv = group[0].cond_dist(*group[0].parents) # Generate a new transformation node
        new_rvs = [new_rv] * len(group)
        vars = Replace(vars, old_rvs, new_rvs)
    transformed_vars = vars
    transformed_given_vars = [transformed_vars[idx] for idx in observed_rv_idx]
    return transformed_vars, transformed_given_vars, given_vals