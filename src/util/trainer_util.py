"""Util functions used by both Leap and Maml which dont have another home.

Consider moving these into subfolders.

Also try move more functions in here."""


import jax
import jax.numpy as np
from jax import vmap
import numpy as npo
import matplotlib.pyplot as plt
import flax

import os
import shutil
from .tensorboard_logger import Logger as TFLogger

import flaxOptimizers
from adahessianJax.flaxOptimizer import Adahessian
from functools import partial

import fenics as fa
import imageio

from . import jax_tools

import pdb

import absl
from absl import app
from absl import flags

FLAGS = flags.FLAGS


def get_ground_truth_points(
    pde, pde_params_list, key, resolution=None, boundary_points=None
):
    """Given a pdedef and list of pde parameters, sample points in the domain and
    evaluate the ground truth at those points."""
    fenics_functions = []
    coefs = []
    coords = []
    keys = jax.random.split(key, len(pde_params_list))

    if resolution is None:
        resolution = FLAGS.ground_truth_resolution

    if boundary_points is None:
        boundary_points = int(
            FLAGS.boundary_resolution_factor * FLAGS.ground_truth_resolution
        )

    for params, key in zip(pde_params_list, keys):
        ground_truth = pde.solve_fenics(
            params, resolution=resolution, boundary_points=boundary_points
        )
        k1, k2 = jax.random.split(key)
        domain_coords = pde.sample_points_in_domain(k1, FLAGS.validation_points, params)
        init_coords = pde.sample_points_initial(k2, FLAGS.validation_points, params)
        fn_coords = np.concatenate([init_coords, domain_coords])
        ground_truth.set_allow_extrapolation(True)
        coefs.append(np.array([ground_truth(x) for x in fn_coords]))
        coords.append(fn_coords)
        ground_truth.set_allow_extrapolation(False)
        fenics_functions.append(ground_truth)
    return fenics_functions, np.stack(coefs, axis=0), np.stack(coords, axis=0)


def extract_coefs_by_dim(function_space, dofs, out, dim_idx=0):
    """Given the value of u (in R^d) at *each* dof coordinate, fill into 'out' the
    vector of dof values corresponding to interpolating u into function_space.

    We need this funtion because each dof corresponds to only one dim of u(x),
    so evaluating u(x) at every dof coordinate gives us d times as many values as
    needed.

    Depending on the function space, dof coordinates might not be repeated (e.g.
    at some x, there is only a dof for some but not all dimensions), and we need to
    use fenics API to extract the right components of u for that dof).
    """
    dofs = dofs.reshape(dofs.shape[0], -1)  # If DoFs is 1d, make it 2d
    if function_space.num_sub_spaces() > 0:
        for sidx in range(function_space.num_sub_spaces()):
            dim_idx = extract_coefs_by_dim(function_space.sub(sidx), dofs, out, dim_idx)
    else:
        if dim_idx >= dofs.shape[1]:
            raise Exception("Mismatch between Fenics fn space dim and dofs arr")
        out[function_space.dofmap().dofs()] = dofs[
            function_space.dofmap().dofs(), dim_idx
        ]
        dim_idx += 1
    return dim_idx


def compare_plots_with_ground_truth(
    model,
    pde,
    fenics_functions,
    params_stacked,
    get_final_model,
    meta_alg_def,
    inner_steps,
):
    """Plot the solutions corresponding to ground truth, and corresponding to the
    meta-learned model after each of k gradient steps.
    """
    keys = jax.random.split(jax.random.PRNGKey(0), len(fenics_functions))
    N = len(fenics_functions)

    params_list = jax_tools.tree_unstack(params_stacked)

    for i in range(min([N, 8])):  # Don't plot more than 8 PDEs for legibility
        ground_truth = fenics_functions[i]

        plt.subplot(inner_steps + 2, min([N, 8]), 1 + i)
        plt.axis("off")
        plt.xlim([FLAGS.xmin * 0.9, FLAGS.xmax * 1.1])
        plt.ylim([FLAGS.ymin * 0.9, FLAGS.ymax * 1.1])
        ground_truth.set_allow_extrapolation(False)
        pde.plot_solution(ground_truth, params_list[i])
        if i == 0:
            plt.title("Truth", fontsize=6)
        for j in range(inner_steps + 1):
            plt.subplot(inner_steps + 2, min([N, 8]), 1 + min([N, 8]) * (j + 1) + i)
            plt.axis("off")

            model, fa_p = model
            final_model = get_final_model(
                keys[i], model, fa_p, params_list[i], j, meta_alg_def,
            )

            coords = ground_truth.function_space().tabulate_dof_coordinates()

            dofs = final_model(np.array(coords))

            out = npo.zeros(len(coords))

            extract_coefs_by_dim(ground_truth.function_space(), dofs, out)

            u_approx = fa.Function(ground_truth.function_space())

            fenics_shape = npo.array(u_approx.vector()).shape
            assert fenics_shape == out.shape, "Mismatched shapes {} and {}".format(
                fenics_shape, out.shape
            )

            u_approx.vector().set_local(out)

            pde.plot_solution(u_approx, params_list[i])
            if i == 0:
                plt.title("NN Model", fontsize=6)

def plot_model_time_series(
    model,
    pde,
    fenics_functions,
    params_stacked,
    get_final_model,
    meta_alg_def,
    inner_steps,
):
    """plot time series gif
    """
    keys = jax.random.split(jax.random.PRNGKey(0), len(fenics_functions))
    N = len(fenics_functions)

    params_list = jax_tools.tree_unstack(params_stacked)

    for i in range(min([N, 8])):  # Don't plot more than 8 PDEs for legibility
        ground_truth = fenics_functions[i]
        tmp_filenames = []
        for t in range(FLAGS.num_tsteps):
            plt.figure()
            plt.subplot(inner_steps + 2, min([N, 8]), 1 + i)
            plt.axis("off")
            plt.xlim([FLAGS.xmin * 0.9, FLAGS.xmax * 1.1])
            plt.ylim([FLAGS.ymin * 0.9, FLAGS.ymax * 1.1])
            ground_truth.set_allow_extrapolation(False)
            t_val = ground_truth.timesteps_list[t]
            pde.plot_solution(ground_truth[t], params_list[i])
            if i == 0:
                plt.title("t = {:.2f} \n Truth".format(t_val), fontsize=6)
            for j in range(inner_steps + 1):
                plt.subplot(inner_steps + 2, min([N, 8]), 1 + min([N, 8]) * (j + 1) + i)
                plt.axis("off")

                #model, fa_p = model
                fa_p = None
                final_model = get_final_model(
                    keys[i], model, fa_p, params_list[i], j, meta_alg_def,
                )

                coords = ground_truth.function_space().tabulate_dof_coordinates()

                # add time dimension
                t_val_expand = np.repeat(t_val, len(coords)).reshape(-1, 1)
                coords_t = np.concatenate([coords, t_val_expand], axis=1)

                dofs = final_model(np.array(coords_t))

                out = npo.zeros(len(coords))

                extract_coefs_by_dim(ground_truth.function_space(), dofs, out)

                u_approx = fa.Function(ground_truth.function_space())

                fenics_shape = npo.array(u_approx.vector()).shape
                assert fenics_shape == out.shape, "Mismatched shapes {} and {}".format(
                    fenics_shape, out.shape
                )

                u_approx.vector().set_local(out)

                pde.plot_solution(u_approx, params_list[i])
                if i == 0:
                    plt.title("NN Model", fontsize=6)
                # save fig here
                plt.savefig('timedependent_burger_' + str(t) + '.png', dpi=400)
                plt.close()
                tmp_filenames.append('timedependent_burger_' + str(t) + '.png')
        #build_gif(tmp_filenames)
    return tmp_filenames


def prepare_logging(out_dir, expt_name):
    if expt_name is not None:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        path = os.path.join(out_dir, expt_name)
        if os.path.exists(path):
            shutil.rmtree(path)
        if not os.path.exists(path):
            os.mkdir(path)

        outfile = open(os.path.join(path, "log.txt"), "w")

        def log(*args, **kwargs):
            print(*args, **kwargs, flush=True)
            print(*args, **kwargs, file=outfile, flush=True)

        tflogger = TFLogger(path)

    else:

        def log(*args, **kwargs):
            print(*args, **kwargs, flush=True)

        tflogger = None

    return path, log, tflogger


@partial(jax.jit, static_argnums=[1, 5])
def vmap_validation_error(
    model, fa_p, ground_truth_params, points, ground_truth_vals, make_coef_func
):
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, FLAGS.n_eval)

    make_coef_func_partial = jax.jit(partial(make_coef_func, model=model, fa_p=fa_p))

    coefs = vmap(make_coef_func_partial, (0, 0, 0))(
        keys, ground_truth_params, points
    )
    coefs = coefs.reshape(coefs.shape[0], coefs.shape[1], -1)
    ground_truth_vals = ground_truth_vals.reshape(coefs.shape)
    err = coefs - ground_truth_vals
    mse = np.mean(err ** 2)
    normalizer = np.mean(ground_truth_vals ** 2, axis=1, keepdims=True)
    rel_sq_err = err ** 2 / normalizer.mean(axis=2, keepdims=True)

    # if contains time dimension, add per time-stepping validation error
    if points.shape[-1] == 3:
        t_rel_sq_err = []
        for i in range(coefs.shape[1] // FLAGS.validation_points):
            t_idx = np.squeeze(np.arange(i * FLAGS.validation_points, (i + 1) * FLAGS.validation_points))
            t_err = err[:, t_idx, :]
            t_normalizer = np.mean(ground_truth_vals[:, t_idx, :] ** 2, axis=1, keepdims=True)
            t_rel_sq_err.append(np.mean(t_err ** 2 / t_normalizer.mean(axis=2, keepdims=True)))

    return (
        mse,
        np.mean(normalizer, axis=(0, 1)),
        np.mean(rel_sq_err),
        np.mean(rel_sq_err, axis=(0, 1)),
        np.std(np.mean(rel_sq_err, axis=(1, 2))),
        np.array(t_rel_sq_err)
    )


def get_optimizer(model_class, init_params):
    if FLAGS.optimizer == "adam":
        optimizer = flax.optim.Adam(learning_rate=FLAGS.outer_lr, beta1=0.8, beta2=0.9).create(
            flax.nn.Model(model_class, init_params)
        )
    elif FLAGS.optimizer == "rmsprop":
            optimizer = flax.optim.Adam(learning_rate=FLAGS.outer_lr, beta1=0., beta2=0.8).create(
                flax.nn.Model(model_class, init_params)
            )
    elif FLAGS.optimizer == "ranger":
        optimizer = flaxOptimizers.Ranger(
            learning_rate=FLAGS.outer_lr, beta2=0.99, use_gc=False
        ).create(flax.nn.Model(model_class, init_params))

    elif FLAGS.optimizer == "adahessian":
        raise Exception("Adahessian currently doesnt work with jitting whole train "
                        "loop or with maml/leap")
        #optimizer = Adahessian(learning_rate=FLAGS.outer_lr, beta2=0.95).create(
        #    flax.nn.Model(model_class, init_params)
        #)
    else:
        raise Exception("unknown optimizer: ", FLAGS.optimizer)

    return optimizer
