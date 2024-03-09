import pangolin
from pangolin import transforms
from pangolin.interface import (
    vmap, normal_scale, normal, RV, VMapDist,
    makerv, viz_upstream, exponential, bernoulli, print_upstream,
    Constant, Index
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


def HashKey(var):
    '''
    var: An RV
    '''
    assert isinstance(var, RV)
    key = (var.cond_dist,)
    for p in var.parents:
        if isinstance(p.cond_dist, Index):
            subkey = (p.parents[0], 'Index')
        elif isinstance(p.cond_dist, Constant):
            subkey = (p.cond_dist.value.shape, 'Constant')
        else:
            subkey = (p, 'Other')
        key = key + subkey
    return key

def Find(vars, given_vars=[]):
    '''
    vars: A list of RVs.
    given_var : A list of RVs that we know is observed
    '''
    assert isinstance(vars, list)
    assert isinstance(given_vars, list)
    HashMap = {}
    for x in vars:
        key = HashKey(x)
        key = key + (x in given_vars,)
        if key in HashMap:
            HashMap[key].append(x)
        else:
            HashMap[key] = [x]
    groupped_rvs = []
    for k, v in HashMap.items():
        if len(v) > 1:
            group_cond_dist = k[0]
            if isinstance(group_cond_dist, Constant):
                # We don't merge constants
                continue
            elif isinstance(group_cond_dist, Index):
                # We don't merge index RV, which causes endless loop
                continue  
            else:
                groupped_rvs.append(v)
    return groupped_rvs


def MergeIndexRVs(vars):
    '''
    Input:
    A list of RVs with Index(slices) as cond_dist, where the value of slices is the same
    for all RVs, i.e. indexed from the same axis.
    In addition, all RVs are indexed from the same RV.
    Lastly, all RVs should be indexed from a single axis only, otherwise vmap cannot work.
    
    Return:
    A single Index RV, and an interger denoting the dimension being indexed.
    '''
    assert isinstance(vars[0].cond_dist, Index)
    assert all(
        x.cond_dist == vars[0].cond_dist for x in vars
    ) # Check all RVs have the same slice, i.e. indexed from the same axis
    assert all(
        x.parents[0] == vars[0].parents[0] for x in vars
    ) # Check all RVs are indexed from the same RV
    slice_config = vars[0].cond_dist.slices
    indexed_dim = None
    for i, s in enumerate(slice_config):
        if s is None:
            if indexed_dim:
                raise ValueError('Multiple indices got indexed')
            indexed_dim = i
    if indexed_dim is None:
        raise ValueError('No indexing dimension found')
    indexed_node = vars[0].parents[0]
    all_indices = np.stack([x.parents[1].cond_dist.value for x in vars]) # Get all indices together
    if np.array_equal(all_indices, np.arange(indexed_node.shape[indexed_dim])):
         # If all dimensions are selected, then no need to create a new index node
        return indexed_node, indexed_dim
    return vars[0].cond_dist(indexed_node, all_indices), indexed_dim


def AggregateParents(parents):
    '''
    parents: A list of RVs denoting the mth parent of a collection of RVs.
    The function would merge these parents into a single RV using different rules
    depending on the their cond_dist
    '''
    cond_dist_type = parents[0].cond_dist
    if isinstance(cond_dist_type, Index):
        # MergeIndexRVs would perform the condition check
        merged_parents, in_axis = MergeIndexRVs(parents) 
    elif isinstance(cond_dist_type, Constant):
        assert all(
            p.cond_dist.value.shape == parents[0].cond_dist.value.shape for p in parents
        ), "Parents are Constant and not having the same shape"
        merged_parents = np.stack([p.cond_dist.value for p in parents])
        in_axis = 0
    else:
        assert all(
            p == parents[0] for p in parents
        ), "Parents not Index not Constant and not being the same RV"
        merged_parents = parents[0]
        in_axis = None
    return merged_parents, in_axis


def Merge(vars, given_vals=[]):
    '''
    vars: A list of RVs that can be vmapped together under the in_axes=None condition.
    given_vals (Optional): The observed values for the RVs in vars
    '''
    assert isinstance(vars, list)
    assert all(
        x.cond_dist == vars[0].cond_dist for x in vars
    ), 'vars not having the same cond_dist'
    if given_vals:
        assert len(given_vals) == len(vars)
        assert (gval.shape == given_vals[0].shape for gval in given_vals)
    base_dist = vars[0].cond_dist
    new_parents = ()
    in_axes = ()
    number_of_parents = len(vars[0].parents)
    for m in range(number_of_parents):
        merged_parents, in_axis = AggregateParents([x.parents[m] for x in vars])
        new_parents =  new_parents + (merged_parents,)
        in_axes = in_axes + (in_axis,)
    vmapped_rv = vmap(base_dist, in_axes, len(vars))(*new_parents)
    if given_vals:
        given_vals = np.stack(given_vals)
    return vmapped_rv, given_vals


def Replace(vars, old, new):
    """
    Given some set of `RV`s, replace some old ones with new ones
    rules: `old` must all on the same level; `old` must not show up in new or
    in the upstream nodes of RVs in new.
    """
    # TODO: Simplify the check
    # upstreams_old = [dag.upstream_nodes(o) for o in old]
    # upstreams_new = [dag.upstream_nodes(n) for n in new]
    # for o in old:
    #     if o in new:
    #         assert False, "old should not show up in new"
    #     if any(o in u_old[:-1] for u_old in upstreams_old):
    #         assert False, "old should all be on the same level in the DAG"
    #     if any(o in u_new[:-1] for u_new in upstreams_new):
    #         assert False, "new should not have old in upstream"

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
            

def AutoVmap(vars, given_vars, given_vals, order=0):
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
    user_input_vars = vars + given_vars
    vars = dag.upstream_nodes(user_input_vars)
    input_vars_idx = [vars.index(var) for var in user_input_vars]
    while True:
        vmappable_groups = Find(vars, given_vars=given_vars)
        if len(vmappable_groups) < 1:
            break
        selected_group = vmappable_groups[order] # We alwayws merge the first group found
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
        vars_updated = Replace(vars, old, new)
        # We need to update the given_vars as well,
        # otherwise RVs in new would never be marked as observed
        for old_var, new_var in zip(vars, vars_updated):
            if old_var in given_vars:
                given_vars[given_vars.index(old_var)] = new_var
        # Input vars in the current DAG
        user_input_vars = [vars_updated[idx] for idx in input_vars_idx] 
        vars = dag.upstream_nodes(vars_updated)
        # Input vars' indices in the updated DAG
        input_vars_idx = [vars.index(var) for var in user_input_vars]
    return [vars[idx] for idx in input_vars_idx], given_vars_t, given_vals_t