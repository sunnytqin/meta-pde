from jax.config import config

import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from jax.experimental import optimizers

from .util.tensorboard_logger import Logger as TFLogger

from .nets import maml

from .poisson import poisson_def
from .nonlinear_stokes import nonlinear_stokes_def

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
    "--lr_inner_lr", type=float, default=1e-1, help="lr for inner learning rate"
)
parser.add_argument(
    "--outer_points",
    type=int,
    default=64,
    help="num query points on the boundary and in domain",
)
parser.add_argument(
    "--inner_points",
    type=int,
    default=64,
    help="num support points on the boundary and in domain",
)
parser.add_argument(
    "--validation_points",
    type=int,
    default=2048,
    help="num points in domain for validation",
)
parser.add_argument("--inner_steps", type=int, default=3, help="num inner steps")
parser.add_argument("--outer_steps", type=int, default=int(1e5), help="num outer steps")
parser.add_argument("--num_layers", type=int, default=3, help="num fcnn layers")
parser.add_argument("--layer_size", type=int, default=64, help="fcnn layer size")
parser.add_argument("--vary_source", type=int, default=1, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=1, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=1, help="1=true.")
parser.add_argument("--siren", type=int, default=0, help="1=true.")
parser.add_argument("--pcgrad", type=float, default=0.0, help="1=true.")
parser.add_argument("--bc_weight", type=float, default=10.0, help="weight on bc loss")
parser.add_argument(
    "--bc_scale", type=float, default=2e-1, help="scale on random uniform bc"
)
parser.add_argument("--pde", type=str, default="poisson", help="which PDE")
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--expt_name", type=str, default="highlr_plot_for_prefpo")
parser.add_argument("--viz_every", type=int, default=1000, help="plot every N steps")
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

    if args.pde == "poisson":
        pde = poisson_def
    elif args.pde == "nonlinear_stokes":
        pde = nonlinear_stokes_def
    else:
        raise Exception("Unknown PDE")

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
    # --------------------- Defining the meta-training algorithm --------------------

    def loss_fn(field_fn, points, params):
        boundary_losses, domain_losses = pde.loss_fn(field_fn, points, params)

        loss = args.bc_weight * np.sum(
            [bl for bl in boundary_losses.values()]
        ) + np.sum([dl for dl in domain_losses.values()])

        # return the total loss, and as aux a dict of individual losses
        return loss, {**boundary_losses, **domain_losses}

    def make_task_loss_fns(key):
        # The input key is terminal
        k1, k2, k3 = jax.random.split(key, 3)
        params = pde.sample_params(k1, args)
        outer_points = pde.sample_points(k2, args.outer_points, params)
        inner_points = pde.sample_points(k2, args.outer_points, params)

        inner_loss = lambda key, field_fn: loss_fn(field_fn, inner_points, params)
        outer_loss = lambda key, field_fn: loss_fn(field_fn, outer_points, params)
        return inner_loss, outer_loss

    make_inner_opt = flax.optim.Momentum(learning_rate=args.inner_lr, beta=0.0).create

    maml_def = maml.MamlDef(
        make_inner_opt=make_inner_opt,
        make_task_loss_fns=make_task_loss_fns,
        inner_steps=args.inner_steps,
        n_batch_tasks=args.bsize,
        softplus_lrs=True,
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

    inner_lr_init, inner_lr_update, inner_lr_get = optimizers.adam(args.lr_inner_lr)

    # Per param per step lrs
    inner_lr_state = inner_lr_init(
        jax.tree_map(
            lambda x: np.stack([np.ones_like(x) for _ in range(args.inner_steps)]),
            optimizer.target,
        )
    )

    # --------------------- Defining the evaluation functions --------------------

    # @partial(jax.jit, static_argnums=(3, 4))
    def get_final_model(key, model_and_lrs, params, inner_steps, maml_def):
        # Input key is terminal
        model, inner_lrs = model_and_lrs
        k1, k2 = jax.random.split(key, 2)
        inner_points = pde.sample_points(k1, args.inner_points, params)
        inner_loss_fn = lambda key, field_fn: loss_fn(field_fn, inner_points, params)

        inner_lrs = jax.tree_map(lambda x: x[:inner_steps], inner_lrs)

        final_model = jax.lax.cond(
            inner_steps != 0,
            lambda _: maml.single_task_rollout(
                maml_def, k2, model, inner_loss_fn, inner_lrs
            )[0],
            lambda _: model,
            0,
        )
        return final_model

    @partial(jax.jit, static_argnums=(4, 5))
    def make_coef_func(key, model_and_lrs, params, coords, inner_steps, maml_def):
        # Input key is terminal
        final_model = get_final_model(key, model_and_lrs, params, inner_steps, maml_def)

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
            maml_def.inner_steps,
            maml_def,
        )

        return np.sqrt(np.mean((coefs - ground_truth_vals.reshape(coefs.shape)) ** 2))

    @jax.jit
    def validation_losses(model_and_lrs, maml_def=maml_def):
        model, inner_lrs = model_and_lrs
        _, losses, meta_losses = maml.multi_task_grad_and_losses(
            maml_def, jax.random.PRNGKey(0), model, inner_lrs,
        )
        return losses, meta_losses

    grid = npo.zeros([101, 101, 2])
    for i in range(101):
        for j in range(101):
            grid[i, j] += [i * 1.0 / 100, j * 1.0 / 100]
    grid = np.array(grid, dtype=np.float32).reshape(-1, 2) * 2 - np.array(
        [[1.0, 1.0]], dtype=np.float32
    )

    assert args.n_eval % 2 == 0

    key, gt_key, gt_points_key = jax.random.split(key, 3)

    gt_keys = jax.random.split(gt_key, args.n_eval)
    gt_params = vmap(pde.sample_params, (0, None))(gt_keys, args)
    print("gt_params: {}".format(gt_params))

    fenics_functions, fenics_vals, coords = trainer_util.get_ground_truth_points(
        args, pde, jax_tools.tree_unstack(gt_params), gt_points_key
    )

    # --------------------- Run MAML --------------------

    for step in range(args.outer_steps):
        key, subkey = jax.random.split(key, 2)

        inner_lrs = inner_lr_get(inner_lr_state)

        with Timer() as t:
            meta_grad, losses, meta_losses = maml.multi_task_grad_and_losses(
                maml_def, subkey, optimizer.target, inner_lrs,
            )
            meta_grad_norm = np.sqrt(
                jax.tree_util.tree_reduce(
                    lambda x, y: x + y,
                    jax.tree_util.tree_map(lambda x: np.sum(x ** 2), meta_grad),
                )
            )
            if np.isfinite(meta_grad_norm):
                if meta_grad_norm > min([100.0, i]):
                    log("clipping gradients with norm {}".format(meta_grad_norm))
                    meta_grad = jax.tree_util.tree_map(
                        lambda x: x / meta_grad_norm, meta_grad
                    )
                optimizer = optimizer.apply_gradient(meta_grad[0])
                inner_lr_state = inner_lr_update(step, meta_grad[1], inner_lr_state)
            else:
                log("NaN grad!")

        val_error = vmap_validation_error(
            (optimizer.target, inner_lrs), gt_params, coords, fenics_vals,
        )

        val_losses, val_meta_losses = validation_losses((optimizer.target, inner_lrs))

        log(
            "step: {}, meta_loss: {}, val_meta_loss: {}, val_err: {}, "
            "meta_grad_norm: {}, time: {}".format(
                step,
                np.mean(meta_losses[0]),
                np.mean(val_meta_losses[0]),
                val_error,
                meta_grad_norm,
                t.interval,
            )
        )
        log(
            "meta_loss_max: {}, meta_loss_min: {}, meta_loss_std: {}".format(
                np.max(meta_losses[0]), np.min(meta_losses[0]), np.std(meta_losses[0])
            )
        )
        log(
            "per_step_losses: {}\nper_step_val_losses:{}\n".format(
                np.mean(losses[0], axis=0), np.mean(val_losses[0], axis=0),
            )
        )

        if tflogger is not None:
            tflogger.log_histogram("batch_meta_losses", meta_losses[0], step)
            tflogger.log_histogram("batch_val_losses", val_meta_losses[0], step)
            tflogger.log_scalar("meta_loss", float(np.mean(meta_losses[0])), step)
            tflogger.log_scalar("val_loss", float(np.mean(val_meta_losses[0])), step)
            for k in meta_losses[1]:
                tflogger.log_scalar(
                    "meta_" + k, float(np.mean(meta_losses[1][k])), step
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

                for inner_step in range(args.inner_steps):
                    for k, v in jax_tools.dict_flatten(inner_lrs.params):
                        tflogger.log_histogram(
                            "inner_lr_{}: ".format(inner_step) + k,
                            jax.nn.softplus(v[inner_step].flatten()),
                            step,
                        )
        if args.viz_every > 0 and step % args.viz_every == 0:
            plt.figure()
            # pdb.set_trace()
            trainer_util.compare_plots_with_ground_truth(
                (optimizer.target, inner_lrs), pde, fenics_functions,
                gt_params, get_final_model, maml_def
            )

            if tflogger is not None:
                tflogger.log_plots("Ground truth comparison", [plt.gcf()], step)

            if args.expt_name is not None:
                plt.savefig(os.path.join(path, "viz_step_{}.png".format(step)), dpi=800)
            else:
                plt.show()

    if args.expt_name is not None:
        outfile.close()

    plt.figure()
    compare_plots_with_ground_truth(
        (optimizer.target, inner_lrs),
        pde,
        fenics_functions,
        gt_params,
        get_final_model,
        leap_def,
    )
    if args.expt_name is not None:
        plt.savefig(os.path.join(path, "viz_final.png"), dpi=800)
    else:
        plt.show()
