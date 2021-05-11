"""Fit NN to one PDE."""
from jax.config import config

import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from jax.experimental import optimizers

from .util.tensorboard_logger import Logger as TFLogger

from .nets import maml
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
parser.add_argument("--bsize", type=int, default=1, help="batch size (in tasks)")
parser.add_argument("--n_eval", type=int, default=2, help="num eval tasks")
parser.add_argument("--outer_lr", type=float, default=1e-5, help="outer learning rate")
parser.add_argument(
    "--outer_points",
    type=int,
    default=1024,
    help="num support points on the boundary and in domain",
)
parser.add_argument(
    "--validation_points",
    type=int,
    default=2048,
    help="num points in domain for validation",
)
parser.add_argument("--sqrt_loss", type=int, default=0, help="1=true. if true, "
                    "minimize the rmse instead of the mse")
parser.add_argument("--outer_steps", type=int, default=int(1e5), help="num outer steps")
parser.add_argument("--num_layers", type=int, default=3, help="num fcnn layers")
parser.add_argument("--layer_size", type=int, default=256, help="fcnn layer size")
parser.add_argument("--vary_source", type=int, default=1, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=1, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=1, help="1=true.")
parser.add_argument("--siren", type=int, default=1, help="1=true.")
parser.add_argument("--pcgrad", type=float, default=0.0, help="1=true.")
parser.add_argument("--bc_weight", type=float, default=100.0, help="weight on bc loss")
parser.add_argument(
    "--bc_scale", type=float, default=1., help="scale on random uniform bc"
)
parser.add_argument("--pde", type=str, default="linear_stokes", help="which PDE")
parser.add_argument("--out_dir", type=str, default=None)
parser.add_argument("--expt_name", type=str, default="nn_default")
parser.add_argument("--viz_every", type=int, default=100, help="plot every N steps")
parser.add_argument("--measure_grad_norm_every", type=int, default=100, help="plot every N steps")

parser.add_argument(
    "--fixed_num_pdes",
    type=int,
    default=1,
    help="set to e.g. 1 to force just 1 possible pde param",
)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = args.pde + "_nn_results"
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
    # --------------------- Defining the meta-training algorithm --------------------

    log(str(args))


    def loss_fn(field_fn, points, params):
        boundary_losses, domain_losses = pde.loss_fn(field_fn, points, params)

        loss = args.bc_weight * np.sum(
            np.array([bl for bl in boundary_losses.values()])
        ) + np.sum(np.array([dl for dl in domain_losses.values()]))

        if args.sqrt_loss:
            loss = np.sqrt(loss)
        # return the total loss, and as aux a dict of individual losses
        return loss, {**boundary_losses, **domain_losses}


    def task_loss_fn(key, model):
        # The input key is terminal
        k1, k2 = jax.random.split(key, 2)
        params = pde.sample_params(k1, args)
        points = pde.sample_points(k2, args.outer_points, params)
        return loss_fn(model, points, params)

    @jax.jit
    def batch_loss_fn(key, model):
        keys = jax.random.split(key, args.bsize)
        losses, aux = jax.vmap(task_loss_fn, (0, None))(keys, model)
        return np.sum(losses), {k: np.sum(v) for k, v in aux.items()}

    def get_grad_norms(key, model):
        _, loss_dict = batch_loss_fn(key, model)
        losses_and_grad_norms = {}
        for k in loss_dict:
            single_loss_fn = lambda model: batch_loss_fn(key, model)[1][k]
            loss_val, loss_grad = jax.value_and_grad(single_loss_fn)(model)
            loss_grad_norm = np.sqrt(
                jax.tree_util.tree_reduce(
                    lambda x, y: x + y,
                    jax.tree_util.tree_map(lambda x: np.sum(x ** 2), loss_grad),
                )
            )
            losses_and_grad_norms[k] = (float(loss_val), float(loss_grad_norm))
        return losses_and_grad_norms

    Field = pde.BaseField.partial(
        sizes=[args.layer_size for _ in range(args.num_layers)],
        dense_args=(),
        nonlinearity=np.sin if args.siren else nn.swish,
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    _, init_params = Field.init_by_shape(subkey, [((1, 2), np.float32)])
    optimizer = flax.optim.Adam(learning_rate=args.outer_lr, beta2=0.98).create(
        flax.nn.Model(Field, init_params)
    )

    # --------------------- Defining the evaluation functions --------------------

    # @partial(jax.jit, static_argnums=(3, 4))
    def get_final_model(key, model, *args):
        # Input key is terminal
        return model

    @partial(jax.jit, static_argnums=(4, 5))
    def make_coef_func(key, model, params, coords, *args):
        # Input key is terminal
        final_model = get_final_model(key, model, params, *args)

        return np.squeeze(final_model(coords))

    @jax.jit
    def vmap_validation_error(
        model, ground_truth_params, points, ground_truth_vals,
    ):
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, args.n_eval)

        coefs = vmap(make_coef_func, (0, None, 0, 0))(
            keys, model, ground_truth_params, points,
        )
        gt = ground_truth_vals.reshape(coefs.shape)

        # this is testing if just not comparing pressures will help
        # todo(alex): put this everywhere or remove it
        if len(coefs.shape) > 2 and coefs.shape[2] > 2:
            coefs = coefs[:, :, :2]
            gt = gt[:, :, :2]

        return np.sqrt(np.mean((coefs - gt) ** 2))

    @jax.jit
    def validation_losses(model):
        return task_loss_fn(jax.random.PRNGKey(0), model)[0]

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
        key, subkey = jax.random.split(key)
        with Timer() as t:
            (loss, loss_aux), batch_grad = jax.value_and_grad(
                batch_loss_fn, argnums=1, has_aux=True
            )(subkey, optimizer.target)

            # ---- This big section is logging a bunch of debug stats
            # loss grad norms; plotting the sampled points; plotting the vals at those
            # points; plotting the losses at those points.

            # Todo (alex) -- see if we can clean it up, and maybe also do it in maml etc
            if args.measure_grad_norm_every > 0 and step % args.measure_grad_norm_every == 0:
                loss_vals_and_grad_norms = get_grad_norms(subkey, optimizer.target)
                print("loss vals and grad norms: ", loss_vals_and_grad_norms)
                if tflogger is not None:
                    for k in loss_vals_and_grad_norms:
                        tflogger.log_scalar("grad_norm_{}".format(k),
                                            float(loss_vals_and_grad_norms[k][1]),
                                            step)
                    _k1, _k2 = jax.random.split(jax.random.split(subkey, args.bsize)[0],
                                                2)
                    _params = pde.sample_params(_k1, args)
                    _points = pde.sample_points(_k2, args.outer_points, _params)
                    plt.figure()
                    for _pointsi in _points:
                        plt.scatter(_pointsi[:, 0], _pointsi[:, 1])
                    tflogger.log_plots("Points", [plt.gcf()], step)
                    _all_points = np.concatenate(_points)
                    _vals = optimizer.target(_all_points)
                    _vals = _vals.reshape((_vals.shape[0], -1))
                    _boundary_losses, _domain_losses = jax.vmap(
                        lambda x: pde.loss_fn(
                            optimizer.target,
                            (x.reshape(1, -1) for _ in range(len(_points))),
                            _params))(_all_points)
                    _all_losses = {**_boundary_losses, **_domain_losses}
                    for _losskey in _all_losses:
                        #print(_losskey)
                        plt.figure()
                        _loss = _all_losses[_losskey]
                        #print(_loss.shape)
                        while len(_loss.shape) > 1:
                            _loss = _loss.mean(axis=1)
                        clrs = plt.scatter(_all_points[:, 0], _all_points[:, 1],
                                           c=_loss[:len(_all_points)])
                        plt.colorbar(clrs)
                        tflogger.log_plots("{}".format(_losskey),
                                           [plt.gcf()], step)

                    for dim in range(_vals.shape[1]):
                        plt.figure()
                        clrs = plt.scatter(_all_points[:, 0], _all_points[:, 1],
                                           c=_vals[:, dim])
                        plt.colorbar(clrs)
                        tflogger.log_plots("Outputs dim {}".format(dim), [plt.gcf()], step)

            if step < 250:
                batch_grad = jax.tree_util.tree_map(
                    lambda x: x * (step+1) / 250,
                    batch_grad
                )
            grad_norm = np.sqrt(
                jax.tree_util.tree_reduce(
                    lambda x, y: x + y,
                    jax.tree_util.tree_map(lambda x: np.sum(x ** 2), batch_grad),
                )
            )
            if np.isfinite(grad_norm):
                if grad_norm > min([100.0, step]):
                    log("clipping gradients with norm {}".format(grad_norm))
                    batch_grad = jax.tree_util.tree_map(
                        lambda x: x / grad_norm, batch_grad
                    )
                optimizer = optimizer.apply_gradient(batch_grad)
            else:
                log("NaN grad!")


        val_error = vmap_validation_error(
            optimizer.target, gt_params, coords, fenics_vals,
        )

        val_loss = validation_losses(optimizer.target)

        log(
            "step: {}, loss: {}, val_loss: {}, val_err: {}, "
            "grad_norm: {}, time: {}".format(
                step, loss, val_loss, val_error, grad_norm, t.interval,
            )
        )

        if tflogger is not None:
            # A lot of these names have unnecessary "meta_"
            # just for consistency with maml_pde and leap_pde
            tflogger.log_scalar("meta_loss", float(np.mean(loss)), step)
            tflogger.log_scalar("val_loss", float(np.mean(val_loss)), step)

            tflogger.log_scalar("val_error", float(val_error), step)
            tflogger.log_scalar("meta_grad_norm", float(grad_norm), step)
            tflogger.log_scalar("step_time", t.interval, step)
            for k in loss_aux:
                tflogger.log_scalar("meta_" + k, float(np.mean(loss_aux[k])), step)


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
                None,
                0,
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
    trainer_util.compare_plots_with_ground_truth(
        optimizer.target, pde, fenics_functions, gt_params, get_final_model, None,
        0,
    )
    if args.expt_name is not None:
        plt.savefig(os.path.join(path, "viz_final.png"), dpi=800)
    else:
        plt.show()
