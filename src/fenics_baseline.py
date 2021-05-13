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

from absl import app
from absl import flags

FLAGS = flags.FLAGS

# parser = argparse.ArgumentParser()
flags.DEFINE_string(
    "test_resolutions",
    "1,2,4,6",
    "mesh resolutions for fenics baseline. expect comma sep ints",
)


def main(argv):
    if FLAGS.out_dir is None:
        FLAGS.out_dir = FLAGS.pde + "_nn_results"

    pde = get_pde(FLAGS.pde)

    path, log, tflogger = trainer_util.prepare_logging(FLAGS.out_dir, FLAGS.expt_name)

    log(str(FLAGS))

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

    test_resolutions = [int(s) for s in ",".split(FLAGS.test_resolutions)]

    def validation_error(
        ground_truth_vals, test_vals,
    ):
        test_vals = test_vals.reshape(test_vals.shape[0], test_vals.shape[1], -1)
        ground_truth_vals = ground_truth_vals.reshape(test_vals.shape)
        err = test_vals - ground_truth_vals
        gt_normalizer = np.mean(ground_truth_vals ** 2, axis=1, keepdims=True)
        test_normalizer = np.mean(ground_truth_vals ** 2, axis=1, keepdims=True)

        return (err ** 2, normalizer, test_normalizer)  # n_pdes x n_points x n_dims

    errs = {}
    times = {}

    for res in test_resolutions:
        with Timer() as t:
            test_fns, test_vals, test_coords = trainer_util.get_ground_truth_points(
                pde,
                jax_tools.tree_unstack(gt_params),
                gt_points_key,
                resolution=args.res,
                boundary_points=int(FLAGS.boundary_resolution_factor * res),
            )
        assert np.allclose(test_coords, coords)
        errs[res] = npo.array(validation_error(ground_truth_vals, test_vals))
        times[res] = t.interval / FLAGS.n_eval

    npo.save(os.path.join(path, "errors_by_resolution.npy"), (errs, times))

    pdb.set_trace()


if __name__ == "__main__":
    app.run(main)
