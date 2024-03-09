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



D = 5
x = [normal_scale(1, 1) for _ in range(D)]
h1 = [normal_scale(sum(x[:i] + x[i + 1:]), 1) for i in range(D)]
h2 = [normal_scale(sum(h1[:i] + h1[i + 1:]), 1) for i in range(D)]

observation = np.random.randn(5,)

calc = Calculate("numpyro",niter=1000)
begin = time.time()
samples = calc.sample(x, h2, observation)
end = time.time()
print(end - begin)