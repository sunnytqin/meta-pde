"""Fit NN to one PDE."""
from jax.config import config

import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from .nets import maml
from .get_pde import get_pde

from functools import partial
import fenics as fa

from .util.timer import Timer

from .util import jax_tools

from .util import trainer_util

import matplotlib.pyplot as plt
import pdb
import sys
import os
import shutil
from copy import deepcopy
from collections import namedtuple

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n_eval", type=int, default=2, help="num eval tasks")

parser.add_argument(
    "--validation_points",
    type=int,
    default=1024,
    help="num points in domain for validation",
)

parser.add_argument("--vary_source", type=int, default=1, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=1, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=1, help="1=true.")

parser.add_argument("--test_resolutions",
                    type=str, default='1,2,4,8,12,16',
                    help="mesh resolutions for fenics baseline. expect comma sep ints")
parser.add_argument("--ground_truth_resolution",
                    type=int, default=20,
                    help="mesh resolution for fenics ground truth")

parser.add_argument("--boundary_resolution_factor",
                    type=float, default=4.,
                    help="ratio of resolution to points around boundary of shape")

parser.add_argument(
    "--bc_scale", type=float, default=1.0, help="scale on random uniform bc"
)
parser.add_argument("--pde", type=str, default="poisson", help="which PDE")

parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--expt_name", type=str, default="nn_default")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.pde + "_baseline_results"
    # make into a hashable, immutable namedtuple
    args = namedtuple("ArgsTuple", vars(args))(**vars(args))

    pde = get_pde(args.pde)

    if args.expt_name is not None:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        path = os.path.join(args.out_dir, args.expt_name)
        if os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.mkdir(path)

        outfile = open(os.path.join(path, "log.txt"), "w")

        def log(*args, **kwargs):
            print(*args, **kwargs, flush=True)
            print(*args, **kwargs, file=outfile, flush=True)
    else:

        def log(*args, **kwargs):
            print(*args, **kwargs, flush=True)

    log(str(args))

    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    key, gt_key, gt_points_key = jax.random.split(key, 3)

    gt_keys = jax.random.split(gt_key, args.n_eval)
    gt_params = vmap(pde.sample_params, (0, None))(gt_keys, args)
    print("gt_params: {}".format(gt_params))

    gt_functions, gt_vals, coords = trainer_util.get_ground_truth_points(
        args, pde, jax_tools.tree_unstack(gt_params), gt_points_key,
        resolution=args.ground_truth_resolution,
        boundary_points=int(args.boundary_resolution_factor*args.ground_truth_resolution)
    )

    test_resolutions = [int(s) for s in ','.split(args.test_resolutions)]

    def validation_error(
        ground_truth_vals, test_vals,
    ):
        test_vals = test_vals.reshape(test_vals.shape[0], test_vals.shape[1], -1)
        ground_truth_vals = ground_truth_vals.reshape(test_vals.shape)
        err = test_vals - ground_truth_vals
        gt_normalizer = np.mean(ground_truth_vals ** 2, axis=1, keepdims=True)
        test_normalizer = np.mean(ground_truth_vals ** 2, axis=1, keepdims=True)

        return (
            err**2, # n_pdes x n_points x n_dims
            normalizer,
            test_normalizer
        )

    errs = {}

    for res in test_resolutions:
        test_fns, test_vals, test_coords = trainer_util.get_ground_truth_points(
            args, pde, jax_tools.tree_unstack(gt_params), gt_points_key,
            resolution=args.res,
            boundary_points=int(args.boundary_resolution_factor*res)
        )
        assert np.allclose(test_coords, coords)
        errs[res] = npo.array(validation_error(ground_truth_vals, test_vals))

    npo.save(os.path.join(path, 'errors_by_resolution.npy'),
             errs)

    pdb.set_trace()
