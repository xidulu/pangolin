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
from jax import numpy as jnp
from itertools import groupby
import time
import numpy as np
from collections import OrderedDict


def _is_the_same(node1, node2):
    # Simplify things a little bit?
    def _check_constant_value_equal(v1, v2):
        if hasattr(v1, 'cond_dist'):
            return np.all(v1.cond_dist.value == v2.cond_dist.value)
        else:
            return np.all(v1 == v2)
    if isinstance(node1.cond_dist, Constant) and isinstance(node2.cond_dist, Constant):
        if node1.shape:
            return all(_check_constant_value_equal(v1, v2)
                       for v1, v2 in zip(node1.cond_dist.value, node2.cond_dist.value))
        return node1.cond_dist.value == node2.cond_dist.value
    return node1 == node2

def merge_if_same(my_list):
    my_list_flattened = jax.tree_util.tree_flatten(my_list)[0]
    if all(
        _is_the_same(element, my_list_flattened[0]) for element in my_list_flattened
    ):
        return my_list_flattened[0], True
    if all(isinstance(element.cond_dist, Constant) for element in my_list_flattened):
        # Assume each element is not nested
        my_list = makerv([element.cond_dist.value for element in my_list_flattened])
    return my_list, False



def get_constructor(node):
    return lambda *args: node.cond_dist(*args)


def process_parents_for_vmap(parents):
    in_axes = ()
    args = []
    for p in parents:
        if isinstance(p, list) or isinstance(p, tuple):
            axis_size = len(p)
            p, merged = merge_if_same(p)
            in_axes = in_axes + (None,) if merged else in_axes + (0,)
            # p = list(p)
        elif isinstance(p.cond_dist, VMapDist):
            in_axes = in_axes + (0,)
            axis_size = p.shape[0]
        args.append(p)
    return args, in_axes, axis_size


def find_common_parent(nodes):
    # Assume nodes are flattened
    first_node = nodes[0]
    num_args = len(first_node.parents)
    for i, p in enumerate(first_node.parents):
        if all(
            p == n.parents[i] for n in nodes
        ):
            return p
    return None

def group_nodes(nodes, group_axis=None):
    nodes_flattened, _ = jax.tree_util.tree_flatten(nodes)
    first_node = nodes_flattened[0]
    assert all(
        str(n.cond_dist) == str(first_node.cond_dist) for n in nodes_flattened
    )
    # Then we group the nodes
    if group_axis is None:
        for i, p in enumerate(first_node.parents):
            if not isinstance(p.cond_dist, Constant):
                group_axis = i
                break
    grouped_nodes = [list(v) for k, v in groupby(nodes_flattened, lambda x: x.parents[group_axis])]
    return grouped_nodes, group_axis


def get_parents(nodes):
    parent_list = list(zip(*[node.parents for node in nodes]))
    parent_list = [list(p) for p in parent_list]
    vmap_idx = None
    for idx, pl in enumerate(parent_list):
        if pl[0].parents:
            vmap_idx = idx
    return parent_list, vmap_idx # this is little bit weird


def _is_grouppable(nodes):
    node_idx = list(range(len(nodes)))
    augmented_node_list = [(idx, n) for idx, n in zip(node_idx, nodes)]
    grouped_result = [
        list(v) for k, v in groupby(augmented_node_list, lambda X: X[1])
    ]
    if len(grouped_result) == len(augmented_node_list):
        return None
    else:
        return [[idx for idx, node in group] for group in grouped_result]


def group_by_lowest_ancestor(nodes):
    grouppable = _is_grouppable(nodes)
    if grouppable:
        return grouppable
    for parent in get_parents(nodes)[0]:
        parent_grouppable = group_by_lowest_ancestor(parent)
        if parent_grouppable:
            return parent_grouppable
    return None


def tuple_if_list(x):
    if isinstance(x, list):
        return tuple(x)
    return x

class AutoVmap():
    def __init__(self, return_vars=[]):
        # We need a mapping from the keys to the index in the list
        # The key would be changing during transformation
        # Everytime 
        assert isinstance(return_vars, list)
        self.return_vars = []
        self._return_var_idx = {}
        if return_vars:
            for i, var in enumerate(return_vars):
                flat_vars, vars_treedef = jax.tree_util.tree_flatten(var)
                self.return_vars.append(flat_vars)
                self._return_var_idx[tuple_if_list(flat_vars)] = i 
                # Initialize the dictionary using their origin values

    def _update_nodes(self, old, new):
        if tuple_if_list(old) in self._return_var_idx:
            node_idx = self._return_var_idx[tuple_if_list(old)]
            self.return_vars[node_idx] = new
            self._return_var_idx[tuple_if_list(new)] = node_idx

    def list_to_vmap(self, nodes):
        if isinstance(nodes, tuple):
            nodes = list(nodes)
        if not isinstance(nodes, list):
            return nodes
        constructor = get_constructor(nodes[0])
        parent_args = zip(*[node.parents for node in nodes])
        # Merge identical parents if possible,
        # For identical parent, in_axes=None, otherwise in_axes=0
        args, in_axes, axis_size = process_parents_for_vmap(parent_args)
        # Re-construct the nodes using vmap
        vmapped_node = vmap(constructor, in_axes, axis_size)(*args)
        return vmapped_node

    def vmap_parent_and_merge(self, nodes):
        if isinstance(nodes, tuple):
            nodes = list(nodes)
        if isinstance(nodes, RV):
            return nodes
        common_parent = find_common_parent(nodes)
        if common_parent:
            new_node = self.list_to_vmap(nodes)
        else:
            parents, vmap_idx = get_parents(nodes)
            parents[vmap_idx] = self.vmap_parent_and_merge(parents[vmap_idx])
            args, in_axes, axis_size = process_parents_for_vmap(parents)
            new_node = vmap(get_constructor(nodes[0]), in_axes, axis_size)(*args)
        self._update_nodes(nodes, new_node)
        return new_node

    def flatten_group_and_merge(self, nodes):
        nodes = jax.tree_util.tree_flatten(nodes)[0]
        group_idx = group_by_lowest_ancestor(nodes)
        if not group_idx:
            return nodes
        nodes_grouped = [[nodes[idx] for idx in group] for group in group_idx]
        nodes_merged = [self.vmap_parent_and_merge(group) for group in nodes_grouped]
        self._update_nodes(nodes, nodes_merged)
        return nodes_merged

    def __call__(self, nodes):
        while True:
            nodes_transformed = self.flatten_group_and_merge(nodes)
            if nodes == nodes_transformed:
                break
            nodes = nodes_transformed
        
        return self.return_vars