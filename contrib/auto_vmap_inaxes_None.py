import pangolin
from pangolin import transforms
from pangolin.interface import (
    vmap, normal_scale, normal, RV, VMapDist,
    makerv, viz_upstream, exponential, bernoulli, print_upstream,
    Constant
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

def get_parents(var):
    '''
    By default, Pangolin's Constant RV parents cannot be compared
    by value diretly. Therefore we need to extract their cond_dist
    for comparision, such that
    x = normal_scale(0, 1)
    y1, y2 = normal_scale(x, 1), normal_scale(x, 1)
    get_parents(y1) == get_parents(y2)
    '''
    parents = []
    for p in var.parents:
        # Need to extract the value for Constant parent,
        if isinstance(p.cond_dist, Constant):
            parents.append(p.cond_dist)
        else:
            parents.append(p)
    return tuple(parents)


def Find(vars, given_vars=[]):
    '''
    vars: A list of RVs.
    given_var : A list of RVs that we know is observed
    '''
    assert isinstance(vars, list)
    assert isinstance(given_vars, list)
    HashMap = {}
    for x in vars:
        key = (x.cond_dist, get_parents(x), x in given_vars)
        if key in HashMap:
            HashMap[key].append(x)
        else:
            HashMap[key] = [x]
    groupped_rvs = []
    for k, v in HashMap.items():
        if len(v) > 1:
            groupped_rvs.append(v)
    return groupped_rvs


def Merge(vars, given_vals=[]):
    '''
    vars: A list of RVs that can be vmapped together under the in_axes=None condition.
    given_vals (Optional): The observed values for the RVs in vars
    '''
    assert isinstance(vars, list)
    assert all(
        get_parents(x) == get_parents(vars[0]) for x in vars
    )
    assert all(
        x.cond_dist == vars[0].cond_dist for x in vars
    )
    if given_vals:
        assert len(given_vals) == len(vars)
        assert (gval.shape == given_vals[0].shape for gval in given_vals)
    base_dist = vars[0].cond_dist
    parents = vars[0].parents
    vmapped_rv = vmap(base_dist, None, len(vars))(*parents)
    if given_vals:
        given_vals = np.stack(given_vals)
    return vmapped_rv, given_vals


def Replace(vars, old, new):
    """
    Given some set of `RV`s, replace some old ones with new ones
    rules: `old` must all on the same level; `old` must not show up in new or
    in the upstream nodes of RVs in new.
    """
    upstreams_old = [dag.upstream_nodes(o) for o in old]
    upstreams_new = [dag.upstream_nodes(n) for n in new]
    for o in old:
        if o in new:
            assert False, "old should not show up in new"
        if any(o in u_old[:-1] for u_old in upstreams_old):
            assert False, "old should all be on the same level in the DAG"
        if any(o in u_new[:-1] for u_new in upstreams_new):
            assert False, "new should not have old in upstream"

    all_vars = dag.upstream_nodes(vars)
    replacements = dict(zip(old, new))

    old_to_new = {}
    for var in all_vars:
        if var in replacements:
            new_var = replacements[var]
        else:
            new_pars = tuple(old_to_new[p] for p in var.parents)
            if new_pars == var.parents:
                new_var = var
            else:
                new_var = RV(var.cond_dist, *new_pars)

        old_to_new[var] = new_var
    return [old_to_new[v] for v in vars]
            

def AutoVmap(vars, given_vars, given_vals):
    '''
    vars: A list of RVs
    given_vars: A list of RVs that we know is observed
    given_vals: The observed value for the RVs in given_vars
    '''
    assert isinstance(vars, list)
    assert isinstance(given_vars, list)
    assert isinstance(given_vals, list)
    assert len(given_vars) == len(given_vals)
    given_vars_t, given_vals_t = [], []
    while True:
        vmappable_groups = Find(vars, given_vars=given_vars)
        if len(vmappable_groups) < 1:
            break
        selected_group = vmappable_groups[0] # We alwayws merge the first group found
        group_is_observed = selected_group[0] in given_vars
        if group_is_observed:
            y_vals = [
                given_vals[given_vars.index(x)] for x in selected_group
            ]
            X_prime, Y = Merge(selected_group, y_vals)
            given_vars_t.append(X_prime)
            given_vals_t.append(Y)
        else:
            X_prime, _ = Merge(selected_group)
        old = selected_group
        new = [X_prime[g] for g in range(len(selected_group))]
        vars_new = Replace(vars, old, new)
        # We need to update the given_vars as well,
        # otherwise RVs in new would never be marked as observed
        for old_var, new_var in zip(vars, vars_new):
            if old_var in given_vars:
                given_vars[given_vars.index(old_var)] = new_var
        vars = vars_new
    return vars, given_vars_t, given_vals_t