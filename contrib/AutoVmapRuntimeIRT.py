import pangolin
from pangolin import transforms
from pangolin.interface import (
    vmap, normal_scale, normal, RV, VMapDist,
    makerv, viz_upstream, exponential, bernoulli, print_upstream,
    Constant, Index, exp, bernoulli_logit
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

data = np.load('./irt_test_data.npy')
num_doctors = 7

def AutoVmapWrap(vars, given_vars=[], given_vals=[], order=-1):
    vars, tree_def = tree_flatten(vars)
    given_vars = tree_flatten(given_vars)[0]
    given_vals = tree_flatten(given_vals)[0]
    transformed_var, transformed_given_var, transformed_given_vals = AutoVmap(
        vars, given_vars, given_vals, order
    )
    return transformed_var, transformed_given_var, transformed_given_vals


def get_standard_model(factor, seed=1):
    y_simulated = data[0, :factor, :, :].reshape(-1, num_doctors)
    num_cases = y_simulated.shape[0]

    doctor_thetas = [normal_scale(0, 2) for _ in range(num_doctors)]
    case_beta = [normal_scale(0, 2) for _ in range(num_cases)]
    case_d = [exp(normal_scale(0.5, 1)) for _ in range(num_cases)]
    response_rvs = []
    response_values = []
    for case_id in range(num_cases):
        for doctor_id in range(num_doctors):  
            d = case_d[case_id]
            beta = case_beta[case_id]
            theta = doctor_thetas[doctor_id]
            response_rvs.append(bernoulli_logit(d * (theta - beta)))
            response_values.append(y_simulated[case_id, doctor_id])
    return [doctor_thetas], response_rvs, response_values
    

def get_manual_vmap_model(factor, seed=1):
    y_simulated = data[0, :factor, :, :].reshape(-1, num_doctors)
    num_cases = y_simulated.shape[0]

    doctor_thetas = vmap(normal_scale, in_axes=None, axis_size=num_doctors)(0, 2)
    case_beta = vmap(normal_scale, in_axes=None, axis_size=num_cases)(0, 2) 
    case_log_d = vmap(normal_scale, in_axes=None, axis_size=num_cases)(0.5, 1) 
    logits = vmap(lambda theta, log_d: exp(log_d) * (doctor_thetas - theta))(case_beta, case_log_d)
    response_rvs = vmap(vmap(bernoulli_logit))(logits)
    return [doctor_thetas], response_rvs, y_simulated


def get_autovmap_model(factor, seed=1):
    vars, given_vars, given_vals = get_standard_model(factor, seed)
    tvars, tgvars, tgvals = AutoVmapWrap(vars, given_vars, given_vals)
    return tvars, tgvars, tgvals

def get_autovmap_transformation_time(factor, seed=1):
    vars, given_vars, given_vals = get_standard_model(factor, seed)
    begin = time.time()
    tvars, tgvars, tgvals = AutoVmapWrap(vars, given_vars, given_vals, -1)
    end = time.time()
    return end - begin


def get_runtime(model_creator, save_dir_template):
    # for K in [1, 2, 5, 10, 20, 50, 100, 200]:
    # for K in [1, 2, 5, 10]:
    # for K in [1, 2, 5, 10]:
    for K in [5, 10]:
        print(f'Testing {model_creator}, with K={K}')
        times = []
        # for seed in range(5):
        for seed in range(1):
            calc = Calculate("numpyro",niter=1000)
            vars, given_vars, given_vals = model_creator(K, seed)
            begin = time.time()
            samples = calc.sample(vars, given_vars, given_vals)∫
            end = time.time()
            times.append(end - begin)
        np.save(save_dir_template.format(K), np.array(times))
    

# get_runtime(get_autovmap_model, './results/irt_auto_vmap__factor_{}.npy')
# get_runtime(get_manual_vmap_model, './results/irt_manual_v˜map__factor_{}.npy')

get_runtime(get_standard_model, './results/irt_no_vmap__factor_{}.npy')

# for K in [1, 2, 5, 10]:
#     save_dir_template = './results/auto_vmap_transform_time_irt__factor_{}.npy'
#     print(f'Getting autovmap transformation time, with K={K}')
#     times = []
#     for seed in range(5):
#         runtime = get_autovmap_transformation_time(K, seed)
#         times.append(runtime)
#     np.save(save_dir_template.format(K), np.array(times))