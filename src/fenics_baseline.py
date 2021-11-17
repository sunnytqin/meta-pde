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
    "test_resolutions",
    "1,2,4,6,8,10,12,16",
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

    gt_functions, gt_vals, coords = trainer_util.get_ground_truth_points(
        pde,
        jax_tools.tree_unstack(gt_params),
        gt_points_key,
        resolution=FLAGS.ground_truth_resolution,
        boundary_points=int(
            FLAGS.boundary_resolution_factor * FLAGS.ground_truth_resolution
        ),
    )

    test_resolutions = [int(s) for s in FLAGS.test_resolutions.split(',')]

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
        if FLAGS.pde == 'td_burgers':
            t_rel_sq_err = []
            tile_idx = test_vals.shape[1] // FLAGS.num_tsteps
            for i in range(FLAGS.num_tsteps):
                t_idx = np.arange(i * tile_idx, (i + 1) * tile_idx)
                t_err = err[:, t_idx, :]
                t_normalizer = np.mean(ground_truth_vals[:, t_idx, :] ** 2, axis=1, keepdims=True)
                t_rel_sq_err.append(np.mean(t_err ** 2 / t_normalizer.mean(axis=2, keepdims=True)))
        else:
            t_rel_sq_err = None

        assert len(err.shape) == 3
        return (
            mse,
            np.mean(normalizer, axis=(0, 1)),
            rel_err,
            per_dim_rel_err,
            rel_err_std,
            t_rel_sq_err,
        )

    errs = {}
    times = {}


    for res in test_resolutions:
        #os.system('dijitso clean')
        print('resolution: ')
        with Timer() as t:
            test_fns, test_vals, test_coords = trainer_util.get_ground_truth_points(
                pde,
                jax_tools.tree_unstack(gt_params),
                gt_points_key,
                resolution=res,
                boundary_points=int(FLAGS.boundary_resolution_factor * res),
            )

        #plt.figure(figsize=(5, 5))
        #fa.plot(test_fns[0].function_space().mesh())

        #u, p = test_fns[0].split()
        #plt.figure(figsize=(9, 3))
        #clrs = fa.plot(u)
        #plt.colorbar(clrs)

        #plt.figure(figsize=(9, 3))
        #fa.plot(u)
        #plt.show()

        #plt.figure(figsize=(5, 5))
        #print('test_fn', type(test_fns[0]))
        #fa.plot(test_fns[0][-1])
        #plt.show()
        #assert np.allclose(test_coords, coords)

        mse, norms, rel_err, per_dim_rel_err, rel_err_std, t_rel_sq_err = validation_error(test_vals, gt_vals)

        log(
            "res: {}, val_mse: {}, "
            "val_rel_err: {}, val_rel_err_std: {}, val_true_norms: {}, "
            "per_dim_rel_err: {}, per_time_step_error: {} ,time: {}".format(
                res,
                mse,
                rel_err,
                rel_err_std,
                norms,
                per_dim_rel_err,
                t_rel_sq_err,
                t.interval,
            )
        )

        errs[res] = validation_error(gt_vals, test_vals)
        times[res] = t.interval / FLAGS.n_eval

    npo.save(os.path.join(path, "errors_by_resolution.npy"), (errs, times), allow_pickle=True)

    #pdb.set_trace()

    for res in test_resolutions:
        log("res: {}, mse: {}, rel_mse: {}, std_rel_mse: {}, per_dim_rel_mse: {}, t_rel_sq_err: {}, time: {}".format(
            res,
            (errs[res][0]).astype(float),
            (errs[res][2]).astype(float),
            (errs[res][4]).astype(float),
            npo.array(errs[res][3]),
            npo.array(errs[res][5]),
            times[res]
        ))

        # errs = (err ** 2, gt_normalizer, rel_err, rel_err_std, per_dim_rel_err)

    #pdb.set_trace()


if __name__ == "__main__":
    app.run(main)
