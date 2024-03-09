import pangolin
from pangolin import transforms
from pangolin.interface import (
    vmap, normal_scale, normal, RV, VMapDist,
    makerv, viz_upstream, exponential, bernoulli, print_upstream,
    Constant
)
from pangolin import Calculate
from pangolin import new_infer
from pangolin import dag
import numpy as np
from contrib.auto_vmap_inaxes_None import Replace, Merge, Find, AutoVmap
from jax.tree_util import tree_flatten, tree_unflatten
from jax import vmap as jvmap
from numpy.testing import assert_allclose
import jax
jax.config.update("jax_enable_x64", True)

def check_same_joint_cond_dist(
        old_var, new_var,
        old_given_var=None, old_given_val=None,
        new_given_var=None, new_given_val=None,
    ):
    '''
    The function requires:
    old_var and new_var having the **PyTree sturcture** and each element
    old_var[i] having the same shape as new_var[i].
    There is no requirement for the given vars and values, as calc.sample would raise
    error if anything goes wrong.

    The function would:
    1. Generate num_samples samples from p(old_var | old_given_var=old_given_val) and
        p(new_var | new_given_var=new_given_val), denoted as X_old and X_new.
    2. Assert the mean and variance of X_old and X_new are close to each other.
    '''
    if old_given_var is not None:
        print('old_given_var provided')
    if new_given_var is not None:
        print('new_given_var provided')
    def _assert(num_samples):
        print(f'Assert using {num_samples} samples')
        calc = Calculate("numpyro",niter=num_samples)
        samples_old_dag = calc.sample(old_var, old_given_var, old_given_val)
        samples_new_dag = calc.sample(new_var, new_given_var, new_given_val)
        flat_old_samples, _ = tree_flatten(samples_old_dag)
        flat_new_samples, _ = tree_flatten(samples_new_dag)
        # Both of shape (D, num_samples)
        flat_old_samples, flat_new_samples = np.stack(flat_old_samples), np.stack(flat_new_samples)
        print(flat_old_samples.mean(-1), flat_new_samples.mean(-1))
        if not np.allclose(flat_old_samples.mean(-1), flat_new_samples.mean(-1), rtol=1e-1, atol=0):
            return False
        if not np.allclose(
            np.cov(flat_old_samples).flatten(),
            np.cov(flat_new_samples).flatten(),
            rtol=1e-1, atol=0
        ):
            return False
        return True
    for num_samples in [10000, 100000, 1000000, 10000000]:
        if _assert(num_samples):
            print('Assert passed')
            return
    raise AssertionError('old_var and new_var do not have equal mean and cov')

# Case 1:
x = normal(1,1)
y = [normal(x,1) for i in range(5)]
z = [[normal(yi + i, 1) for i in range(6)] for yi in y]
flat_vars, tree_def = tree_flatten([x, y, z])
transformed_var, transformed_given_var, transformed_given_vals = AutoVmap(flat_vars, [], [])
old_var = tree_unflatten(tree_def, flat_vars)
new_var = tree_unflatten(tree_def, transformed_var)
# check_same_joint_cond_dist(old_var, old_var)
check_same_joint_cond_dist(old_var, new_var)

# Case 2
z = normal(0, 1)
x = [normal_scale(z, 1) for _ in range(2)]
y1 = [normal_scale(x[0], 2) for _ in range(3)]
y2 = [normal_scale(3, x[1]) for _ in range(3)]
given_vals = [np.array(i * 1.0) - 5 for i in range(3)]
all_vars, tree_def_allvars = tree_flatten([z, x, y1, y2])
given_vars, tree_def_gvars = tree_flatten(y2)
transformed_var, transformed_given_var, transformed_given_vals = AutoVmap(
    all_vars, given_vars, given_vals
)
old_var = [z, x, y1, y2]
new_var = tree_unflatten(tree_def_allvars, transformed_var)
# This would fail
# check_same_joint_cond_dist(
#     old_var, old_var,
#     y2, given_vals,
#     transformed_given_var, transformed_given_vals
# )
check_same_joint_cond_dist(
    old_var, old_var,
    y2, given_vals,
    y2, given_vals
)

check_same_joint_cond_dist(
    old_var, new_var,
    y2, given_vals,
    transformed_given_var, transformed_given_vals # Both are lists of only one element
)