from pangolin.interface import RV
from pangolin import Calculate
from pangolin import dag
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten
import jax
jax.config.update("jax_enable_x64", True)

def check_same_joint_cond_dist(
        old_var, new_var,
        old_given_var=None, old_given_val=None,
        new_given_var=None, new_given_val=None,
        verbose=False
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
    # TODO: Check inputs have the same PyTree structure
    if verbose:
        if old_given_var is not None:
            print('old_given_var provided')
        if new_given_var is not None:
            print('new_given_var provided')
    def _assert(num_samples):
        if verbose:
            print(f'Assert using {num_samples} samples')
        calc = Calculate("numpyro",niter=num_samples)
        samples_old_dag = calc.sample(old_var, old_given_var, old_given_val)
        samples_new_dag = calc.sample(new_var, new_given_var, new_given_val)
        flat_old_samples, _ = tree_flatten(samples_old_dag)
        flat_new_samples, _ = tree_flatten(samples_new_dag)
        # Both of shape (D, num_samples)
        flat_old_samples, flat_new_samples = np.stack(flat_old_samples), np.stack(flat_new_samples)
        if not np.allclose(flat_old_samples.mean(-1), flat_new_samples.mean(-1), rtol=1e-1, atol=1e-1):
            if verbose:
                print('old_var mean:', flat_old_samples.mean(-1))
                print('new_var mean:', flat_new_samples.mean(-1))
            return False
        if not np.allclose(
            np.cov(flat_old_samples).flatten(),
            np.cov(flat_new_samples).flatten(),
            rtol=1e-1, atol=1e-1
        ):
            if verbose:
                print('old_var cov:', np.cov(flat_old_samples))
                print('new_var cov:', np.cov(flat_new_samples))
            return False
        return True
    for num_samples in [10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000]:
        if _assert(num_samples):
            if verbose:
                print('Assert passed')
            return
    raise AssertionError('old_var and new_var do not have equal mean and cov')


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