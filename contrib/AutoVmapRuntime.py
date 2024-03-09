import pangolin
from pangolin import transforms
from pangolin.interface import (
    vmap, normal_scale, normal, RV, VMapDist,
    makerv, viz_upstream, exponential, bernoulli, print_upstream,
    Constant, Index, exp
)
from pangolin import Calculate
from pangolin import dag
import jax
from jax.random import PRNGKey
from jax.tree_util import tree_map, tree_flatten
from jax import numpy as jnp
import time
import numpy as np
from collections import defaultdict
from contrib import AutoVmap
import time


def get_standard_model(K, seed=1):
    Y_SIZE = 2 ** K
    Z_SIZE = 2 ** K
    x = normal(1,1)
    y = [normal(x + 1,1) for i in range(Y_SIZE)]
    z = [[normal(yi,1) for i in range(Z_SIZE)] for yi in y]
    observed_values = jax.random.normal(PRNGKey(seed), (Y_SIZE * Z_SIZE,))
    return [x], tree_flatten([z])[0], list(observed_values.flatten())


def get_manual_vmap_model(K, seed=1):
    Y_SIZE = 2 ** K
    Z_SIZE = 2 ** K
    x = normal(1, 1)
    y = vmap(normal_scale, (None, None), axis_size=Y_SIZE)(x + 1, 1)
    z = vmap(vmap(normal_scale, (None, None), axis_size=Z_SIZE), (0, None))(y, 1)
    observed_values = jax.random.normal(PRNGKey(seed), (Y_SIZE, Z_SIZE))
    return [x], z, observed_values


def get_autovmap_model(K, seed=1):
    vars, given_vars, given_vals = get_standard_model(K, seed)
    tvars, tgvars, tgvals = AutoVmap(vars, given_vars, given_vals)
    return tvars, tgvars, tgvals

def get_autovmap_transformation_time(K, seed=1):
    vars, given_vars, given_vals = get_standard_model(K, seed)
    begin = time.time()
    tvars, tgvars, tgvals = AutoVmap(vars, given_vars, given_vals)
    end = time.time()
    return end - begin


def get_runtime(model_creator, save_dir_template):
    for K in [1, 2, 3, 4, 5, 6]:
        print(f'Testing {model_creator}, with K={K}')
        times = []
        for seed in range(10):
            calc = Calculate("numpyro",niter=1000)
            vars, given_vars, given_vals = model_creator(K, seed)
            begin = time.time()
            samples = calc.sample(vars, given_vars, given_vals)
            end = time.time()
            times.append(end - begin)
        np.save(save_dir_template.format(K), np.array(times))
    

# get_runtime(get_manual_vmap_model, './results/manual_vmap__K_{}.npy')
# get_runtime(get_autovmap_model, './results/auto_vmap__K_{}.npy')
# get_runtime(get_standard_model, './results/no_vmap__K_{}.npy')

for K in [1, 2, 3, 4, 5, 6]:
    save_dir_template = './results/auto_vmap_transform_time__K_{}.npy'
    print(f'Getting autovmap transformation time, with K={K}')
    times = []
    for seed in range(10):
        runtime = get_autovmap_transformation_time(K, seed)
        times.append(runtime)
    np.save(save_dir_template.format(K), np.array(times))