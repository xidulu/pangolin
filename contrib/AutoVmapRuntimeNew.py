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
import argparse


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--K', default=0.01, type=int)
parser.add_argument('--iter', default=None, type=int)
