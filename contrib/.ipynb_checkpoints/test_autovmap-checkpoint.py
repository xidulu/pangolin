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
from auto_vmap_new import Replace, Merge, EncodeVars, FindVmappable, AutoVmap
from jax.tree_util import tree_flatten


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

    # x = normal_scale(0, 1)
    # y = [normal_scale(x, 1) for _ in range(2)]
    # z = [normal_scale(y[i % 2], 1) for i in range(10)]
    # transformed_rvs = replace([x] + y + z, [x, y[0]], [normal_scale(1, 1), normal_scale(x, 2)])
    # new_x = transformed_rvs[0]
    # new_ys = transformed_rvs[1:3]
    # new_zs = transformed_rvs[3:]
    # for i, new_z in enumerate(new_zs):
    #     assert new_z.parents[0] == new_ys[i % 2]
    # for new_y in new_ys:
    #     assert new_y.parents[0] == new_x
    # assert new_ys[0].parents[1].cond_dist == Constant(2)


def test_merge_rvs():
    x = normal_scale(0, 1)
    y = [normal_scale(x, 1) for _ in range(10)]
    vmapped_y = Merge(y)
    assert vmapped_y.cond_dist.axis_size == 10

def test_encode_vars():
    x = normal_scale(0, 1)
    y = [normal_scale(x, 1) for _ in range(10)]
    y_vals = [np.array(i * 1.0) - 5 for i in range(10)]
    encoded_vars, var_indices = EncodeVars([x], y, y_vals)
    assert len(encoded_vars) == len([x]) + len(y)
    assert encoded_vars[var_indices[0]][0] == x

    x = normal_scale(0, 1)
    y1 = [normal_scale(x, 1) for _ in range(10)]
    y2 = [normal_scale(x, 1) for _ in range(10)]
    y_vals = [np.array(i * 1.0) - 5 for i in range(10)]
    encoded_vars, var_indices = EncodeVars([x] + y1, y2, y_vals)
    assert len(encoded_vars) == len([x] + y1) + len(y2)
    input_vars = [x] + y1
    for i, idx in enumerate(var_indices):
        assert encoded_vars[idx][0] == input_vars[i]


def test_group_rvs():
    z = normal(0, 1)
    x = [normal_scale(z, 1) for _ in range(2)]
    y1 = [normal_scale(x[0], 1) for _ in range(3)]
    y2 = [normal_scale(x[1], 1) for _ in range(3)]
    encoded_vars, _ = EncodeVars(x + y1 + y2, y2, [np.array(i * 1.0) - 5 for i in range(3)])
    grouped_rvs = FindVmappable(encoded_vars)
    assert len(grouped_rvs) == 3

def test_AutoVmap():
    z = normal(0, 1)
    x = [normal_scale(z, 1) for _ in range(2)]
    y1 = [normal_scale(x[0], 1) for _ in range(3)]
    y2 = [normal_scale(x[1], 1) for _ in range(3)]
    given_vals = [np.array(i * 1.0) - 5 for i in range(3)]
    all_vars = [z] + x + y1 + y2
    given_vars = y1
    transformed_var, transformed_given_var, transformed_given_vals = AutoVmap(all_vars, given_vars, given_vals)

    x = normal(0,1)
    y = [normal(x,1) for i in range(5)]
    z = [[normal(yi,1) for i in range(6)] for yi in y]
    all_vars = tree_flatten([x, y, z])[0]
    transformed_var, transformed_given_var, transformed_given_vals = AutoVmap(all_vars, [], [])
    print(transformed_var[-1])


# test_replace()
# test_merge_rvs()
# test_encode_vars()
# test_group_rvs()
test_AutoVmap()