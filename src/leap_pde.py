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


from .util import common_flags

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("bsize", 16, "batch size (in tasks)")
flags.DEFINE_float("outer_lr", 1e-4, "outer learning rate")

flags.DEFINE_float("inner_lr", 1e-5, "inner learning rate")
flags.DEFINE_float("inner_grad_clip", 1., "inner grad clipping")

flags.DEFINE_integer("inner_steps", 10, "num_inner_steps")


def main(argv):
    if FLAGS.out_dir is None:
        FLAGS.out_dir = FLAGS.pde + "_leap_results"

    pde = get_pde(FLAGS.pde)

    path, log, tflogger = trainer_util.prepare_logging(FLAGS.out_dir, FLAGS.expt_name)

    log(FLAGS.flags_into_string())

    # --------------------- Defining the meta-training algorithm --------------------

    def loss_fn(field_fn, points, params):
        boundary_losses, domain_losses = pde.loss_fn(field_fn, points, params)

        loss = FLAGS.bc_weight * np.sum(
            np.array([bl for bl in boundary_losses.values()])
        ) + np.sum(np.array([dl for dl in domain_losses.values()]))

        # return the total loss, and as aux a dict of individual losses
        return loss, {**boundary_losses, **domain_losses}

    def make_task_loss_fn(key):
        # The input key is terminal
        params = pde.sample_params(key)

        def variational_energy_estimator(inner_key, field_fn, params=params):
            points = pde.sample_points(inner_key, FLAGS.inner_points, params)
            return loss_fn(field_fn, points, params)

        return variational_energy_estimator

    make_inner_opt = flax.optim.Momentum(learning_rate=FLAGS.inner_lr, beta=0.0).create

    leap_def = leap.LeapDef(
        make_inner_opt=make_inner_opt,
        make_task_loss_fn=make_task_loss_fn,
        inner_steps=FLAGS.inner_steps,
        n_batch_tasks=FLAGS.bsize,
        norm=True,
        loss_in_distance=True,
        stabilize=True,
    )

    Field = pde.BaseField.partial(
        sizes=[FLAGS.layer_size for _ in range(FLAGS.num_layers)],
        dense_args=(),
        nonlinearity=np.sin if FLAGS.siren else nn.swish,
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    _, init_params = Field.init_by_shape(subkey, [((1, 2), np.float32)])
    optimizer = flax.optim.Adam(learning_rate=FLAGS.outer_lr).create(
        flax.nn.Model(Field, init_params)
    )

    # --------------------- Defining the evaluation functions --------------------

    # @partial(jax.jit, static_argnums=(3, 4))
    def get_final_model(key, model, params, inner_steps, leap_def):
        # Input key is terminal
        k1, k2 = jax.random.split(key, 2)
        inner_points = pde.sample_points(k1, FLAGS.inner_points, params)
        inner_loss_fn = lambda key, field_fn: loss_fn(field_fn, inner_points, params)

        temp_leap_def = leap_def._replace(inner_steps=inner_steps)
        final_model = jax.lax.cond(
            inner_steps != 0,
            lambda _: leap.single_task_rollout(temp_leap_def, k2, model, inner_loss_fn)[
                0
            ],
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
        keys = jax.random.split(key, FLAGS.n_eval)
        coefs = vmap(make_coef_func, (0, None, 0, 0, None, None))(
            keys,
            model_and_lrs,
            ground_truth_params,
            points,
            leap_def.inner_steps,
            leap_def,
        )
        coefs = coefs.reshape(coefs.shape[0], coefs.shape[1], -1)
        ground_truth_vals = ground_truth_vals.reshape(coefs.shape)
        err = coefs - ground_truth_vals
        rel_sq_err = err ** 2 / np.mean(ground_truth_vals ** 2, axis=1, keepdims=True)

        return np.sqrt(np.mean(rel_sq_err)), np.sqrt(np.mean(rel_sq_err, axis=(0, 1)))

    @jax.jit
    def validation_losses(model, leap_def=leap_def):
        meta_grad, losses = leap.multi_task_grad_and_losses(
            leap_def, jax.random.PRNGKey(0), model,
        )
        return losses

    @jax.jit
    def train_step(key, optimizer):
        meta_grad, losses = leap.multi_task_grad_and_losses(
            leap_def, key, optimizer.target,
        )
        meta_grad_norm = np.sqrt(
            jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_util.tree_map(lambda x: np.sum(x ** 2), meta_grad[0]),
            )
        )
        meta_grad = jax.lax.cond(
            meta_grad_norm > FLAGS.grad_clip,
            lambda grad_tree: jax.tree_util.tree_map(
                lambda x: FLAGS.grad_clip * x / meta_grad_norm, grad_tree
            ),
            lambda grad_tree: grad_tree,
            meta_grad
        )
        optimizer = optimizer.apply_gradient(meta_grad)
        return optimizer, losses, meta_grad_norm


    key, gt_key, gt_points_key = jax.random.split(key, 3)

    gt_keys = jax.random.split(gt_key, FLAGS.n_eval)
    gt_params = vmap(pde.sample_params)(gt_keys)
    print("gt_params: {}".format(gt_params))

    fenics_functions, fenics_vals, coords = trainer_util.get_ground_truth_points(
        pde, jax_tools.tree_unstack(gt_params), gt_points_key
    )

    # --------------------- Run LEAP --------------------

    for step in range(FLAGS.outer_steps):
        key, subkey = jax.random.split(key, 2)

        with Timer() as t:
            optimizer, losses, meta_grad_norm = train_step(subkey, optimizer)

        if step % FLAGS.val_every == 0:
            val_error, per_dim_val_error = vmap_validation_error(
                optimizer.target, gt_params, coords, fenics_vals,
            )

            val_losses = validation_losses(optimizer.target)

        if step % FLAGS.log_every == 0:
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
                for i in range(len(per_dim_val_error)):
                    tflogger.log_scalar(
                        "val_error_dim_{}".format(i), float(per_dim_val_error[i]), step
                    )
                for k in losses[1]:
                    tflogger.log_scalar(
                        "meta_" + k, float(np.mean(losses[1][k][:, -1])), step
                    )
                for inner_step in range(FLAGS.inner_steps + 1):
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

                if step % FLAGS.viz_every == 0:
                    # These take lots of filesize so only do them sometimes

                    for k, v in jax_tools.dict_flatten(optimizer.target.params):
                        tflogger.log_histogram("Param: " + k, v.flatten(), step)

        if FLAGS.viz_every > 0 and step % FLAGS.viz_every == 0:
            plt.figure()
            # pdb.set_trace()
            trainer_util.compare_plots_with_ground_truth(
                optimizer.target,
                pde,
                fenics_functions,
                gt_params,
                get_final_model,
                leap_def,
                FLAGS.inner_steps,
            )
            if FLAGS.expt_name is not None:
                plt.savefig(os.path.join(path, "viz_step_{}.png".format(step)), dpi=800)

            if tflogger is not None:
                tflogger.log_plots("Ground truth comparison", [plt.gcf()], step)

    if FLAGS.expt_name is not None:
        outfile.close()

    plt.figure()
    trainer_util.compare_plots_with_ground_truth(
        optimizer.target,
        pde,
        fenics_functions,
        gt_params,
        get_final_model,
        leap_def,
        FLAGS.inner_steps,
    )
    if FLAGS.expt_name is not None:
        plt.savefig(os.path.join(path, "viz_final.png"), dpi=800)
    else:
        plt.show()


if __name__ == "__main__":
    app.run(main)
