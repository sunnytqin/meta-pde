"""Use LEAP to amortize fitting a NN across a class of PDEs."""
from jax.config import config

import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from jax.experimental import optimizers

from .util.tensorboard_logger import Logger as TFLogger

from .nets import leap

from .get_pde import get_pde

from functools import partial
import flax
from flax import nn
import fenics as fa

from .util import pcgrad
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
parser.add_argument("--bsize", type=int, default=16, help="batch size (in tasks)")
parser.add_argument("--n_eval", type=int, default=16, help="num eval tasks")
parser.add_argument("--inner_lr", type=float, default=1e-3, help="inner learning rate")
parser.add_argument("--outer_lr", type=float, default=3e-4, help="outer learning rate")
parser.add_argument(
    "--inner_points",
    type=int,
    default=512,
    help="num support points on the boundary and in domain",
)
parser.add_argument(
    "--validation_points",
    type=int,
    default=1024,
    help="num points in domain for validation",
)
parser.add_argument("--sqrt_loss", type=int, default=0, help="1=true. if true, "
                    "minimize the rmse instead of the mse")
parser.add_argument("--inner_steps", type=int, default=10, help="num inner steps")
parser.add_argument("--outer_steps", type=int, default=int(1e5), help="num outer steps")
parser.add_argument("--num_layers", type=int, default=3, help="num fcnn layers")
parser.add_argument("--layer_size", type=int, default=128, help="fcnn layer size")
parser.add_argument("--vary_source", type=int, default=1, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=1, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=1, help="1=true.")
parser.add_argument("--siren", type=int, default=0, help="1=true.")
parser.add_argument("--pcgrad", type=float, default=0.0, help="1=true.")
parser.add_argument("--bc_weight", type=float, default=100., help="weight on bc loss")
parser.add_argument(
    "--bc_scale", type=float, default=2e-1, help="scale on random uniform bc"
)
parser.add_argument("--grad_clip", type=float, default=None, help="max grad for clipping")

parser.add_argument("--pde", type=str, default="linear_stokes", help="which PDE")
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--expt_name", type=str, default="leap_default")
parser.add_argument("--viz_every", type=int, default=1000, help="plot every N steps")
parser.add_argument("--val_every", type=int, default=25, help="validate every N steps")

parser.add_argument(
    "--fixed_num_pdes",
    type=int,
    default=None,
    help="set to e.g. 1 to force just 1 possible pde param",
)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.pde + "_meta_results"
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

        tflogger = TFLogger(path)

    else:

        def log(*args, **kwargs):
            print(*args, **kwargs, flush=True)

        tflogger = None

    log(str(args))

    # --------------------- Defining the meta-training algorithm --------------------

    def loss_fn(field_fn, points, params):
        boundary_losses, domain_losses = pde.loss_fn(field_fn, points, params)

        loss = args.bc_weight * np.sum(
            np.array([bl for bl in boundary_losses.values()])
        ) + np.sum(np.array([dl for dl in domain_losses.values()]))

        if args.sqrt_loss:
            loss = np.sqrt(loss)
        # return the total loss, and as aux a dict of individual losses
        return loss, {**boundary_losses, **domain_losses}

    def make_task_loss_fn(key):
        # The input key is terminal
        params = pde.sample_params(key, args)

        def variational_energy_estimator(inner_key, field_fn, params=params):
            points = pde.sample_points(inner_key, args.inner_points, params)
            return loss_fn(field_fn, points, params)

        return variational_energy_estimator

    make_inner_opt = flax.optim.Momentum(learning_rate=args.inner_lr, beta=0.0).create

    leap_def = leap.LeapDef(
        make_inner_opt=make_inner_opt,
        make_task_loss_fn=make_task_loss_fn,
        inner_steps=args.inner_steps,
        n_batch_tasks=args.bsize,
        norm=True,
        loss_in_distance=True,
        stabilize=True,
    )

    Field = pde.BaseField.partial(
        sizes=[args.layer_size for _ in range(args.num_layers)],
        dense_args=(),
        nonlinearity=np.sin if args.siren else nn.swish,
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    _, init_params = Field.init_by_shape(subkey, [((1, 2), np.float32)])
    optimizer = flax.optim.Adam(learning_rate=args.outer_lr).create(
        flax.nn.Model(Field, init_params)
    )

    # --------------------- Defining the evaluation functions --------------------

    # @partial(jax.jit, static_argnums=(3, 4))
    def get_final_model(key, model, params, inner_steps, leap_def):
        # Input key is terminal
        k1, k2 = jax.random.split(key, 2)
        inner_points = pde.sample_points(k1, args.inner_points, params)
        inner_loss_fn = lambda key, field_fn: loss_fn(field_fn, inner_points, params)

        temp_leap_def = leap_def._replace(inner_steps=inner_steps)
        final_model = jax.lax.cond(
            inner_steps != 0,
            lambda _: leap.single_task_rollout(temp_leap_def, k2, model,
                                               inner_loss_fn)[0],
            lambda _: model,
            0,
        )
        return final_model

    @partial(jax.jit, static_argnums=(4, 5))
    def make_coef_func(key, model_and_lrs, params, coords, inner_steps, leap_def):
        # Input key is terminal
        final_model = get_final_model(key, model_and_lrs, params, inner_steps, leap_def)

        return np.squeeze(final_model(coords))

    @jax.jit
    def vmap_validation_error(
        model_and_lrs, ground_truth_params, points, ground_truth_vals,
    ):
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, args.n_eval)
        coefs = vmap(make_coef_func, (0, None, 0, 0, None, None))(
            keys,
            model_and_lrs,
            ground_truth_params,
            points,
            leap_def.inner_steps,
            leap_def,
        )
        coefs = coefs.reshape(coefs.shape[0], -1)
        ground_truth_vals = ground_truth_vals.reshape(coefs.shape)
        err = coefs - ground_truth_vals
        rel_sq_err = err**2 / np.mean(ground_truth_vals**2, axis=0, keepdims=True)

        return np.sqrt(np.mean(rel_sq_err))

    @jax.jit
    def validation_losses(model, leap_def=leap_def):
        meta_grad, losses = leap.multi_task_grad_and_losses(
            leap_def, jax.random.PRNGKey(0), model,
        )
        return losses

    assert args.n_eval % 2 == 0

    key, gt_key, gt_points_key = jax.random.split(key, 3)

    gt_keys = jax.random.split(gt_key, args.n_eval)
    gt_params = vmap(pde.sample_params, (0, None))(gt_keys, args)
    print("gt_params: {}".format(gt_params))

    fenics_functions, fenics_vals, coords = trainer_util.get_ground_truth_points(
        args, pde, jax_tools.tree_unstack(gt_params), gt_points_key
    )

    # --------------------- Run LEAP --------------------

    for step in range(args.outer_steps):
        key, subkey = jax.random.split(key, 2)

        with Timer() as t:
            meta_grad, losses = leap.multi_task_grad_and_losses(
                leap_def, subkey, optimizer.target,
            )
            meta_grad_norm = np.sqrt(
                jax.tree_util.tree_reduce(
                    lambda x, y: x + y,
                    jax.tree_util.tree_map(lambda x: np.sum(x ** 2), meta_grad),
                )
            )
            if np.isfinite(meta_grad_norm):
                if args.grad_clip is not None and meta_grad_norm > args.grad_clip:
                    log("clipping gradients with norm {}".format(meta_grad_norm))
                    meta_grad = jax.tree_util.tree_map(
                        lambda x: args.grad_clip*x / meta_grad_norm, meta_grad
                    )
                optimizer = optimizer.apply_gradient(meta_grad)
            else:
                log("NaN grad!")
        if step % args.val_every == 0:
            val_error = vmap_validation_error(
                optimizer.target, gt_params, coords, fenics_vals,
            )

            val_losses = validation_losses(optimizer.target)

        log(
            "step: {}, meta_loss: {}, val_meta_loss: {}, val_err: {}, "
            "meta_grad_norm: {}, time: {}".format(
                step,
                np.mean(losses[0][:, -1]),
                np.mean(val_losses[0][:, -1]),
                val_error,
                meta_grad_norm,
                t.interval,
            )
        )
        log(
            "meta_loss_max: {}, meta_loss_min: {}, meta_loss_std: {}".format(
                np.max(losses[0][:, -1]),
                np.min(losses[0][:, -1]),
                np.std(losses[0][-1]),
            )
        )
        log(
            "per_step_losses: {}\nper_step_val_losses:{}\n".format(
                np.mean(losses[0], axis=0), np.mean(val_losses[0], axis=0),
            )
        )

        if tflogger is not None:
            tflogger.log_histogram("batch_meta_losses", losses[0][:, -1], step)
            tflogger.log_histogram("batch_val_losses", val_losses[0][:, -1], step)
            tflogger.log_scalar("meta_loss", float(np.mean(losses[0][:, -1])), step)
            tflogger.log_scalar("val_loss", float(np.mean(val_losses[0][:, -1])), step)
            for k in losses[1]:
                tflogger.log_scalar(
                    "meta_" + k, float(np.mean(losses[1][k][:, -1])), step
                )
            for inner_step in range(args.inner_steps + 1):
                tflogger.log_scalar(
                    "loss_step_{}".format(inner_step),
                    float(np.mean(losses[0][:, inner_step])),
                    step,
                )
                tflogger.log_scalar(
                    "val_loss_step_{}".format(inner_step),
                    float(np.mean(val_losses[0][:, inner_step])),
                    step,
                )
                tflogger.log_histogram(
                    "batch_loss_step_{}".format(inner_step),
                    losses[0][:, inner_step],
                    step,
                )
                tflogger.log_histogram(
                    "batch_val_loss_step_{}".format(inner_step),
                    val_losses[0][:, inner_step],
                    step,
                )
                for k in losses[1]:
                    tflogger.log_scalar(
                        "{}_step_{}".format(k, inner_step),
                        float(np.mean(losses[1][k][:, inner_step])),
                        step,
                    )
            tflogger.log_scalar("val_error", float(val_error), step)
            tflogger.log_scalar("meta_grad_norm", float(meta_grad_norm), step)
            tflogger.log_scalar("step_time", t.interval, step)

            if step % args.viz_every == 0:
                # These take lots of filesize so only do them sometimes

                for k, v in jax_tools.dict_flatten(optimizer.target.params):
                    tflogger.log_histogram("Param: " + k, v.flatten(), step)

        if args.viz_every > 0 and step % args.viz_every == 0:
            plt.figure()
            # pdb.set_trace()
            trainer_util.compare_plots_with_ground_truth(
                optimizer.target,
                pde,
                fenics_functions,
                gt_params,
                get_final_model,
                leap_def,
                args.inner_steps,
            )
            if args.expt_name is not None:
                plt.savefig(os.path.join(path, "viz_step_{}.png".format(step)), dpi=800)

            if tflogger is not None:
                tflogger.log_plots("Ground truth comparison", [plt.gcf()], step)



    if args.expt_name is not None:
        outfile.close()

    plt.figure()
    trainer_util.compare_plots_with_ground_truth(
        optimizer.target, pde, fenics_functions, gt_params, get_final_model, leap_def,
        args.inner_steps,
    )
    if args.expt_name is not None:
        plt.savefig(os.path.join(path, "viz_final.png"), dpi=800)
    else:
        plt.show()
