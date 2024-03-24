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
from copy import deepcopy

def del_indices(my_list, indices):
    return [x for i, x in enumerate(my_list) if i not in indices]

def find_first_none_consecutive(my_list):
    '''
    @Author: ChatGPT
    '''
    try:
        # Step 1: Find the index of the first None
        first_none_index = my_list.index(None)
    except ValueError:
        # None is not in the list
        raise ValueError('No indexed dim found!')

    # Step 2 & 3: Check if None values are consecutive
    for i in range(first_none_index, len(my_list)):
        if my_list[i] != None and i + 1 < len(my_list) and my_list[i + 1] == None:
            # Found a non-None value followed by another None, so they are not consecutive
            return 0

    # Step 4: All None values are consecutive
    return first_none_index



def HashKey(var):
    '''
    var: An RV
    '''
    assert isinstance(var, RV)
    key = (var.cond_dist,)
    for p in var.parents:
        if isinstance(p.cond_dist, Index):
            # Need to ensure all indexed node having the same "slice"
            # however slice is not hashable, so we convert them to string
            subkey = (p.parents[0], str(p.cond_dist.slices), tuple(var.shape for var in p.parents[1:]))
        elif isinstance(p.cond_dist, Constant):
            subkey = (p.cond_dist.value.shape,)
        else:
            subkey = (p,)
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


# def MergeIndexRVs(vars):
#     '''
#     Input:
#     A list of RVs with Index(slices) as cond_dist, where the value of slices is the same
#     for all RVs, i.e. indexed from the same axis.
#     In addition, all RVs are indexed from the same RV.
#     Lastly, all RVs should be indexed from a single axis only, otherwise vmap cannot work.
    
#     Return:
#     A single Index RV, and an interger denoting the dimension being indexed.
#     '''
#     assert isinstance(vars[0].cond_dist, Index)
#     assert all(
#         x.cond_dist == vars[0].cond_dist for x in vars
#     ) # Check all RVs have the same slice, i.e. indexed from the same axis
#     assert all(
#         x.parents[0] == vars[0].parents[0] for x in vars
#     ) # Check all RVs are indexed from the same RV
#     slice_config = vars[0].cond_dist.slices
#     indexed_dim = None
#     for i, s in enumerate(slice_config):
#         if s is None:
#             if indexed_dim:
#                 raise ValueError('Multiple indices got indexed')
#             indexed_dim = i
#     if indexed_dim is None:
#         raise ValueError('No indexing dimension found')
#     indexed_node = vars[0].parents[0]
#     all_indices = np.stack([x.parents[1].cond_dist.value for x in vars]) # Get all indices together
#     if np.array_equal(all_indices, np.arange(indexed_node.shape[indexed_dim])):
#          # If all dimensions are selected, then no need to create a new index node
#         return indexed_node, indexed_dim
#     return vars[0].cond_dist(indexed_node, all_indices), indexed_dim


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
    indexed_dims = [i for i, s in enumerate(slice_config) if s is None]
    assert len(indexed_dims) > 0, 'no indexed dim found'
    indexed_node = vars[0].parents[0]
    all_indices = []
    for i in range(len(indexed_dims)):
        assert all(
            var.parents[1:][i].cond_dist.value.shape == vars[0].parents[1:][i].cond_dist.value.shape
            for var in vars
        ), "All indicies should have the same shape"
        all_indices.append(
            np.stack([var.parents[1:][i].cond_dist.value for var in vars])
        )
    # Special case: Only one axis is indexed, and all elements are selected,
    # then we don't need additional index node
    if len(all_indices) == 1:
        assert len(indexed_dims) == 1
        vmap_dim = indexed_dims[0]
        if np.array_equal(all_indices[0], np.arange(indexed_node.shape[vmap_dim])):
            return indexed_node, vmap_dim
    
    # Find vmap dim, if slices are consecutive, then it is dimension of the first "None" in slice,
    # otherwise it is the 0th dimension.
    vmap_dim = find_first_none_consecutive(slice_config)
    return vars[0].cond_dist(indexed_node, *all_indices), vmap_dim


def AggregateParents(parents):
    '''
    parents: A list of RVs denoting the mth parent of a collection of RVs.
    The function would merge these parents into a single RV using different rules
    depending on the their cond_dist
    '''
    cond_dist_type = parents[0].cond_dist
    if all(p == parents[0] for p in parents):
        merged_parents = parents[0]
        in_axis = None
    elif isinstance(cond_dist_type, Index):
        # MergeIndexRVs would perform the condition check
        merged_parents, in_axis = MergeIndexRVs(parents) 
    elif isinstance(cond_dist_type, Constant):
        assert all(
            p.cond_dist.value.shape == parents[0].cond_dist.value.shape for p in parents
        ), "Parents are Constant and not having the same shape"
        merged_parents = np.stack([p.cond_dist.value for p in parents])
        in_axis = 0
    else:
        raise ValueError("Parents cannot be merged together")
        
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


def SimpleReplace(vars, old, new):
    """
    Replace nodes by directly modifying the DAG
    """
    assert all(
        isinstance(x.cond_dist, Index) for x in new
    )
    replacements = dict(zip(old, new))
    apply_replacements = lambda x: replacements[x] if x in replacements else x
    for i in range(len(vars)):
        if vars[i] in replacements:
            vars[i] = replacements[vars[i]]
        else:
            vars[i].parents = [apply_replacements(p) for p in vars[i].parents]
    return vars

            

def AutoVmap(vars, given_vars, given_vals, order=-1):
    '''
    vars: A list of RVs
    given_vars: A list of RVs that we know is observed
    given_vals: The observed value for the RVs in given_vars
    '''
    assert isinstance(vars, list)
    assert isinstance(given_vars, list)
    assert isinstance(given_vals, list)
    assert len(given_vars) == len(given_vals)
    assert all(isinstance(var, RV) for var in vars)
    assert all(isinstance(var, RV) for var in given_vars)
    user_input_vars = vars
    vars = dag.upstream_nodes(vars + given_vars) # All vars in the DAG
    for var in vars:
        var.__dict__['_frozen'] = False
    vars = sorted(vars, key=lambda rv: rv.id)
    input_vars_idx = [vars.index(var) for var in user_input_vars]
    while True:
        vmappable_groups = Find(vars, given_vars=given_vars)
        if len(vmappable_groups) < 1:
            break
        selected_group = vmappable_groups[order]
        group_is_observed = selected_group[0] in given_vars
        if group_is_observed:
            indices = [given_vars.index(x) for x in selected_group]
            y_vals = [
                given_vals[i] for i in indices
            ]
            X_prime, Y = Merge(selected_group, y_vals)
            given_vars.append(X_prime)
            given_vals.append(Y)
            # Remove out-of-date given_vars and given_vals
            given_vars = del_indices(given_vars, indices)
            given_vals = del_indices(given_vals, indices)    
        else:
            X_prime, _ = Merge(selected_group)
        old = selected_group
        new = [X_prime[g] for g in range(len(selected_group))]
        for var in [X_prime] + new:
            var.__dict__['_frozen'] = False
        # vars_updated = Replace(vars, old, new) # vars_update has same length as vars
        vars_updated = SimpleReplace(vars, old, new) # Direclty modify the RVs
        
        # We need to update the given_vars as well, otherwise the RVs in given_vars
        # will be out-of-date after we performed Replace
        for old_var, new_var in zip(vars, vars_updated):
            if old_var in given_vars:
                # given_vars should NEVER contain index node
                assert not isinstance(new_var.cond_dist, Index) 
                given_vars[given_vars.index(old_var)] = new_var
        # Add the newly created VRV to the vars.
        # vars always contains all RVs in the DAG
        vars = vars_updated + [X_prime]
    for var in vars:
        var.__dict__['_frozen'] = True
    return [vars[idx] for idx in input_vars_idx], given_vars, given_vals