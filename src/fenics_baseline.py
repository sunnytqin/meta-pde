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

from .util import jax_tools, trainer_util, common_flags, trainer_util


import matplotlib.pyplot as plt
import pdb
import sys
import os
import shutil
from copy import deepcopy
from collections import namedtuple

import argparse

from absl import app
from absl import flags

import tracemalloc


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "spatial_resolutions",
    "1,2,4,6,8,10,12,16",
    "mesh resolutions for fenics baseline. expect comma sep ints",
)

flags.DEFINE_string(
    "boundary_resolutions",
    "4,16,64,256,1024,4096",
    "mesh resolutions for fenics baseline. expect comma sep ints",
)

flags.DEFINE_string(
    "time_resolutions",
    "1",
    "mesh resolutions for fenics baseline. expect comma sep ints",
)


def main(argv):
    if FLAGS.out_dir is None:
        FLAGS.out_dir = FLAGS.pde + "_fenics_results"

    pde = get_pde(FLAGS.pde)

    path, log, tflogger = trainer_util.prepare_logging(FLAGS.out_dir, FLAGS.expt_name)

    log(FLAGS.flags_into_string())

    key, subkey = jax.random.split(jax.random.PRNGKey(0))
    key, gt_key, gt_points_key = jax.random.split(key, 3)

    gt_keys = jax.random.split(gt_key, FLAGS.n_eval)
    gt_params = vmap(pde.sample_params)(gt_keys)
    print("gt_params: {}".format(gt_params))

    spatial_resolutions = [int(s) for s in FLAGS.spatial_resolutions.split(',')]
    time_resolutions = [int(s) for s in FLAGS.time_resolutions.split(',')]
    if FLAGS.pde == 'td_burgers':
        base_time_resolution = FLAGS.num_tsteps
    else:
        base_time_resolution = 0
    time_resolutions = [int((base_time_resolution - 1) * np.power(2, t) + 1) for t in time_resolutions]
    boundary_resolutions = [int(s) for s in FLAGS.boundary_resolutions.split(',')]

    FLAGS.num_tsteps = int((time_resolutions[-1] - 1) * 2 + 1)
    gt_functions, gt_vals, coords = trainer_util.get_ground_truth_points(
        pde,
        jax_tools.tree_unstack(gt_params),
        gt_points_key,
        resolution=FLAGS.ground_truth_resolution,
        boundary_points=int(
            FLAGS.boundary_resolution_factor * FLAGS.ground_truth_resolution
        ),
    )


    def validation_error(ground_truth_vals, test_vals):
        test_vals = test_vals.reshape(test_vals.shape[0], test_vals.shape[1], -1)
        ground_truth_vals = ground_truth_vals.reshape(test_vals.shape)
        err = test_vals - ground_truth_vals
        mse = np.mean(err ** 2)
        normalizer = np.mean(ground_truth_vals ** 2, axis=1, keepdims=True)
        rel_sq_err = err**2 / normalizer.mean(axis=2, keepdims=True)

        rel_err = np.mean(rel_sq_err)
        rel_err_std = np.std(np.mean(rel_sq_err, axis=(1, 2)))
        per_dim_rel_err = np.mean(rel_sq_err, axis=(0, 1))

        # if contains time dimension, add per time-stepping validation error
        #if FLAGS.pde == 'td_burgers':
        #    t_rel_sq_err = []
        #    tile_idx = test_vals.shape[1] // FLAGS.num_tsteps
        #    for i in range(FLAGS.num_tsteps):
        #        t_idx = np.arange(i * tile_idx, (i + 1) * tile_idx)
        #        t_err = err[:, t_idx, :]
        #        t_normalizer = np.mean(ground_truth_vals[:, t_idx, :] ** 2, axis=1, keepdims=True)
        #        t_rel_sq_err.append(np.mean(t_err ** 2 / t_normalizer.mean(axis=2, keepdims=True)))
        #else:
        #    t_rel_sq_err = None

        assert len(err.shape) == 3
        return (
            mse,
            np.mean(normalizer, axis=(0, 1)),
            rel_err,
            per_dim_rel_err,
            rel_err_std,
            #t_rel_sq_err,
        )

    errs = {}
    times = {}

    for s_res in spatial_resolutions:
        for t_res in time_resolutions:
            for b_res in boundary_resolutions:
                FLAGS.num_tsteps = t_res
                res = s_res
                with Timer() as t:
                    test_fns, test_vals, test_coords = trainer_util.get_ground_truth_points(
                        pde,
                        jax_tools.tree_unstack(gt_params),
                        gt_points_key,
                        #resolution=res,
                        #boundary_points=int(FLAGS.boundary_resolution_factor * res),
                        resolution=s_res,
                        boundary_points=b_res,
                    )

                if FLAGS.pde == 'td_burgers':
                    # this works for fenics function ground_truth
                    exact_vals = []
                    for i in range(test_coords.shape[0]):
                        test_coord = test_coords[i, :, :]
                        exact_val = npo.empty_like(test_vals[i])
                        ground_truth_function = gt_functions[i]
                        for j in range(test_coord.shape[0]):
                            exact_val[j] = ground_truth_function(test_coord[j])
                        exact_vals.append(exact_val)
                    exact_vals = np.array(exact_vals)
                else:
                    exact_vals = gt_vals

                for i, u in enumerate(test_fns):
                    plt.figure(figsize=(5, 5))
                    pde.plot_solution(u)
                    if FLAGS.expt_name is not None:
                        plt.savefig(os.path.join(path, f"eval_{i}_res_{s_res}_{b_res}_{t_res}.png"), dpi=800)

                if FLAGS.pde == 'td_burgers':
                    errs[(s_res, t_res)] = validation_error(exact_vals, test_vals)
                    times[(s_res, t_res)] = t.interval / FLAGS.n_eval
                else:
                    errs[(s_res, b_res)] = validation_error(exact_vals, test_vals)
                    times[(s_res, b_res)] = t.interval / FLAGS.n_eval

    npo.save(os.path.join(path, "errors_by_resolution.npy"), (errs, times), allow_pickle=True)

    if FLAGS.pde == 'td_burgers':
        for s_res in spatial_resolutions:
            for t_res in time_resolutions:
                log("res: {}, mse: {}, rel_mse: {}, std_rel_mse: {}, per_dim_rel_mse: {}, time: {}".format(
                    (s_res, t_res),
                    (errs[(s_res, t_res)][0]).astype(float),
                    (errs[(s_res, t_res)][2]).astype(float),
                    (errs[(s_res, t_res)][4]).astype(float),
                    npo.array(errs[(s_res, t_res)][3]),
                    times[(s_res, t_res)]
                ))
    else:
        for s_res in spatial_resolutions:
            for b_res in boundary_resolutions:
                log("res: {}, mse: {}, rel_mse: {}, std_rel_mse: {}, per_dim_rel_mse: {}, time: {}".format(
                    (s_res, b_res),
                    (errs[(s_res, b_res)][0]).astype(float),
                    (errs[(s_res, b_res)][2]).astype(float),
                    (errs[(s_res, b_res)][4]).astype(float),
                    npo.array(errs[(s_res, b_res)][3]),
                    times[(s_res, b_res)]
                ))


if __name__ == "__main__":
    app.run(main)
