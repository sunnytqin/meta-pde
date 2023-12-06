"""Use MAML to amortize fitting a NN across a class of PDEs."""

from jax.config import config
#config.update('jax_disable_jit', True)

import jax
import jax.numpy as np
import numpy as npo
import re
from jax import grad, jit, vmap

from jax.experimental import optimizers

from .util.tensorboard_logger import Logger as TFLogger

from .nets import maml

from .get_pde import get_pde


from functools import partial
import flax
from flax import nn
import fenics as fa

from .util.timer import Timer

from .util import jax_tools

from .util import trainer_util

import matplotlib.pyplot as plt
from collections import deque
import pdb
import sys
import os
import pickle
import shutil

import time

from .util import common_flags

from absl import app
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_integer("bsize", 16, "batch size (in tasks)")
flags.DEFINE_float("outer_lr", 1e-3, "outer learning rate")

flags.DEFINE_float("inner_lr", 3e-5, "inner learning rate")
flags.DEFINE_float("lr_inner_lr", 1.0 / 2, "lr for inner learning rate")
flags.DEFINE_integer("inner_steps", 5, "num_inner_steps")
flags.DEFINE_float("inner_grad_clip", 1e14, "inner grad clipping")

flags.DEFINE_float("outer_loss_decay", 0.1, "0. = just take final loss. 1.=sum all")


def main(arvg):
    if FLAGS.out_dir is None:
        FLAGS.out_dir = FLAGS.pde + "_maml_results"

    pde = get_pde(FLAGS.pde)

    path, log, tflogger = trainer_util.prepare_logging(FLAGS.out_dir, FLAGS.expt_name)

    log(FLAGS.flags_into_string())

    with open(os.path.join(path, "flags_config.txt"), "w") as f:
        f.write(FLAGS.flags_into_string())

    # --------------------- Defining the meta-training algorithm --------------------

    def loss_fn(field_fn, points, params):
        boundary_losses, domain_losses = pde.loss_fn(field_fn, points, params)
        loss = np.sum(
                FLAGS.bc_weight * 
                np.sum(np.array([bl for bl in boundary_losses.values()])) + 
                np.sum(np.array([dl for dl in domain_losses.values()]))
                )

        # return the total loss, and as aux a dict of individual losses
        return loss, {**boundary_losses, **domain_losses}

    def make_task_loss_fns(key):
        # The input key is terminal
        params = pde.sample_params(key)
        def inner_loss(key, field_fn, params=params):
            inner_points = pde.sample_points(key, FLAGS.inner_points, params)
            return loss_fn(field_fn, inner_points, params)

        def outer_loss(key, field_fn, params=params):
            outer_points = pde.sample_points(key, FLAGS.outer_points, params)
            return loss_fn(field_fn, outer_points, params)

        return inner_loss, outer_loss

    make_inner_opt = flax.optim.Momentum(learning_rate=FLAGS.inner_lr, beta=0.0).create
    #make_inner_opt = flax.optim.Adam(learning_rate=FLAGS.inner_lr, beta1=0.9, beta2=0.99).create

    maml_def = maml.MamlDef(
        make_inner_opt=make_inner_opt,
        make_task_loss_fns=make_task_loss_fns,
        inner_steps=FLAGS.inner_steps,
        n_batch_tasks=FLAGS.bsize,
        softplus_lrs=True,
        outer_loss_decay=FLAGS.outer_loss_decay,
    )

    Field = pde.BaseField.partial(
        sizes=[FLAGS.layer_size for _ in range(FLAGS.num_layers)],
        dense_args=(),
        nonlinearity=np.sin if FLAGS.siren else nn.swish,
        omega=FLAGS.siren_omega,
        omega0=FLAGS.siren_omega0,
        log_scale=FLAGS.log_scale,
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    # Model params
    _, init_params = Field.init_by_shape(subkey, [((1, 2), np.float32)])

    if FLAGS.load_model_from_expt is not None:
        model_path = FLAGS.load_model_from_expt
        model_file = npo.array(
            [f for f in os.listdir(model_path) if "model_step" in f]
        )
        steps = npo.zeros_like(model_file, dtype=int)
        for i, f in enumerate(model_file):
            steps[i] = re.findall('[0-9]+', f)[-1]
        model_file = model_file[
            np.argsort(steps)[-1]
        ]
        log('load pre-trained model from file: ', model_file)
        with open(os.path.join(model_path, model_file), 'rb') as f:
            optimizer_target = pickle.load(f)

        init_params = flax.serialization.from_state_dict(init_params, optimizer_target[0]['params'])

    optimizer = trainer_util.get_optimizer(Field, init_params)

    log(
        'NN model:', jax.tree_map(lambda x: x.shape, init_params)
    )

    # Per param per step lrs
    inner_lr_init, inner_lr_update, inner_lr_get = optimizers.adam(FLAGS.lr_inner_lr, b1=0.9, b2=0.99,)
    lr_init_params = jax.tree_map(
        lambda x: np.stack([np.ones_like(x) for _ in range(FLAGS.inner_steps)]),
        optimizer.target,
    )
    if FLAGS.load_model_from_expt is not None:
        _, value_tree = jax.tree_util.tree_flatten(lr_init_params)
        value_flat, _ = jax.tree_util.tree_flatten(optimizer_target[1]['params'])
        lr_init_params = jax.tree_util.tree_unflatten(value_tree, value_flat)
    inner_lr_state = inner_lr_init(lr_init_params)

    # --------------------- Defining the evaluation functions --------------------

    @partial(jax.jit, static_argnums=(3, 4))
    def get_final_model(key, model_and_lrs, params, inner_steps, maml_def):
        # Input key is terminal
        model, inner_lrs = model_and_lrs
        k1, k2 = jax.random.split(key, 2)
        inner_points = pde.sample_points(k1, FLAGS.inner_points, params)
        inner_loss_fn = lambda key, field_fn: loss_fn(field_fn, inner_points, params)

        inner_lrs = jax.tree_map(lambda x: x[:inner_steps], inner_lrs)

        temp_maml_def = maml_def._replace(inner_steps=inner_steps)

        final_model = jax.lax.cond(
            inner_steps != 0,
            lambda _: maml.single_task_rollout(
                temp_maml_def, k2, model, inner_loss_fn, inner_lrs
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

    partial_make_coef_func = lambda key, model_and_lrs, params, coords: make_coef_func(
        key, model_and_lrs, params, coords, maml_def.inner_steps, maml_def)

    @jax.jit
    def validation_losses(model_and_lrs, maml_def=maml_def):
        model, inner_lrs = model_and_lrs
        _, losses, meta_losses = maml.multi_task_grad_and_losses(
            maml_def, jax.random.PRNGKey(0), model, inner_lrs,
        )
        return losses, meta_losses

    @jax.jit
    def train_step(step, key, optimizer, inner_lr_state):
        inner_lrs = inner_lr_get(inner_lr_state)

        meta_grad, losses, meta_losses = maml.multi_task_grad_and_losses(
            maml_def, key, optimizer.target, inner_lrs,
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
        optimizer = optimizer.apply_gradient(meta_grad[0])
        inner_lr_state = inner_lr_update(step, meta_grad[1], inner_lr_state)
        return optimizer, inner_lr_state, losses, meta_losses, meta_grad_norm

    key, gt_key, gt_points_key = jax.random.split(key, 3)

    gt_keys = jax.random.split(gt_key, FLAGS.n_eval)
    gt_params = vmap(pde.sample_params)(gt_keys)
    print("gt_params: {}".format(gt_params))

    fenics_functions, fenics_vals, coords = trainer_util.get_ground_truth_points(
        pde, jax_tools.tree_unstack(gt_params), gt_points_key
    )

    if FLAGS.pde == 'td_burgers':
        t_list = fenics_functions[0].timesteps_list

    time_last_log = time.time()
    # --------------------- Run MAML --------------------

    for step in range(FLAGS.outer_steps):
        key, subkey = jax.random.split(key, 2)

        with Timer() as t:
            (optimizer, inner_lr_state, losses, meta_losses, meta_grad_norm) = train_step\
                    (step, subkey, optimizer, inner_lr_state)
            inner_lrs = inner_lr_get(inner_lr_state)

        if (
                FLAGS.measure_grad_norm_every > 0
                and step % FLAGS.measure_grad_norm_every == 0
        ):
            meta_losses_report = {k: np.sum(v).astype(float) for k, v in meta_losses[1].items()}
            log("loss vals and grad norms: ", meta_losses_report)
            log("meta grad norm: ", meta_grad_norm)

        if np.isnan(np.mean(meta_losses[0])):
            log("encountered nan at at step {}".format(step))
            break

        if step % FLAGS.log_every == 0:
            with Timer() as deploy_timer:
                mse, norms, rel_err, per_dim_rel_err, rel_err_std, t_rel_sq_err = trainer_util.vmap_validation_error(
                    (optimizer.target, inner_lrs), gt_params, coords,
                    fenics_vals,
                    partial_make_coef_func)
                mse.block_until_ready()
            deployment_time = deploy_timer.interval / FLAGS.n_eval

            val_losses, val_meta_losses = validation_losses(
                (optimizer.target, inner_lrs)
            )

            log(
                "step: {}, meta_loss: {}, val_meta_loss: {}, val_mse: {}, "
                "val_rel_err: {}, val_rel_err_std: {}, val_true_norms: {}, "
                "per_dim_rel_err: {}, per_time_step_error: {}, deployment_time: {},"
                "meta_grad_norm: {}, time: {}, key: {}, subkey: {}".format(
                    step,
                    np.mean(meta_losses[0]),
                    np.mean(val_meta_losses[0]),
                    mse,
                    rel_err,
                    rel_err_std,
                    norms,
                    per_dim_rel_err,
                    t_rel_sq_err,
                    deployment_time,
                    meta_grad_norm,
                    t.interval,
                    key,
                    subkey,
                )
            )
            if step > 0:
                log("time {} steps: {}".format(FLAGS.log_every,
                                               time.time() - time_last_log))
                time_last_log = time.time()
            log(
                "meta_loss_max: {}, meta_loss_min: {}, meta_loss_std: {}".format(
                    np.max(meta_losses[0]), np.min(meta_losses[0]), np.std(meta_losses[0])
                )
            )
            log(
                "per_step_losses: {}\nper_step_val_losses:{}\n".format(
                    np.mean(losses, axis=0), np.mean(val_losses, axis=0),
                )
            )

            if tflogger is not None:
                tflogger.log_histogram("batch_meta_losses", meta_losses[0], step)
                tflogger.log_histogram("batch_val_losses", val_meta_losses[0], step)
                tflogger.log_scalar("meta_loss", float(np.mean(meta_losses[0])), step)
                tflogger.log_scalar("val_loss", float(np.mean(val_meta_losses[0])), step)
                tflogger.log_scalar("val_rel_mse", float(rel_err), step)
                tflogger.log_scalar("std_val_rel_mse", float(rel_err_std), step)

                tflogger.log_scalar("val_mse", float(mse), step)

                for i in range(len(per_dim_rel_err)):
                    tflogger.log_scalar(
                        "val_rel_error_dim_{}".format(i), float(per_dim_rel_err[i]), step
                    )
                    tflogger.log_scalar("val_norm_dim_{}".format(i), float(norms[i]), step)
                for k in meta_losses[1]:
                    tflogger.log_scalar(
                        "meta_" + k, float(np.mean(meta_losses[1][k])), step
                    )

                tflogger.log_scalar("meta_grad_norm", float(meta_grad_norm), step)
                tflogger.log_scalar("step_time", t.interval, step)

                if step % FLAGS.viz_every == 0:
                    # These take lots of filesize so only do them sometimes
                    if FLAGS.pde == 'td_burgers':
                        plt.figure()
                        plt.plot(t_list, t_rel_sq_err, '.')
                        plt.xlabel('t')
                        plt.ylabel('val rel err')
                        tflogger.log_plots(
                            "Per time step relative error", [plt.gcf()], step
                        )

                    plt.figure()
                    plt.plot(np.arange(FLAGS.inner_steps + 1), np.mean(losses, axis=0))
                    plt.xlabel('Inner Step')
                    plt.ylabel('Loss')
                    tflogger.log_plots(
                        "Per inner step step loss", [plt.gcf()], step
                    )

                    plt.figure()
                    plt.plot(np.arange(FLAGS.inner_steps + 1), np.mean(val_losses, axis=0))
                    plt.xlabel('Inner Step')
                    plt.ylabel('Val Loss')
                    tflogger.log_plots(
                        "Per inner step step val loss", [plt.gcf()], step
                    )

                    for inner_step in range(FLAGS.inner_steps + 1):
                        # tflogger.log_scalar(
                        #    "loss_inner_step_{}".format(inner_step),
                        #    float(np.mean(losses[0][:, inner_step])),
                        #    step,
                        # )
                        # tflogger.log_scalar(
                        #    "val_loss_inner_step_{}".format(inner_step),
                        #    float(np.mean(val_losses[0][:, inner_step])),
                        #    step,
                        # )
                        tflogger.log_histogram(
                            "batch_loss_inner_step_{}".format(inner_step),
                            losses[:, inner_step],
                            step,
                        )
                        tflogger.log_histogram(
                            "batch_val_loss_inner_step_{}".format(inner_step),
                            val_losses[:, inner_step],
                            step,
                        )
                        # for k in losses[1]:
                        #    tflogger.log_scalar(
                        #        "{}_inner_step_{}".format(k, inner_step),
                        #        float(np.mean(losses[1][k][:, inner_step])),
                        #        step,
                        #    )

                    for k, v in jax_tools.dict_flatten(optimizer.target.params):
                        tflogger.log_histogram("Param: " + k, v.flatten(), step)

                    for inner_step in range(FLAGS.inner_steps):
                        try:
                            for k, v in jax_tools.dict_flatten(inner_lrs.params):
                                tflogger.log_histogram(
                                    "inner_lr_{}: ".format(inner_step) + k,
                                    jax.nn.softplus(v[inner_step].flatten()),
                                    step,
                                )
                        except:
                            print("inner lr", inner_lrs[inner_step])
        if FLAGS.viz_every > 0 and step % FLAGS.viz_every == 0:
            if FLAGS.pde == 'td_burgers':
                plt.figure()
                trainer_util.plot_model_time_series_new(
                    (optimizer.target, inner_lrs),
                    pde,
                    fenics_functions,
                    gt_params,
                    get_final_model,
                    maml_def,
                    FLAGS.inner_steps,
                )
            else:
                plt.figure()
                # pdb.set_trace()
                trainer_util.compare_plots_with_ground_truth(
                    (optimizer.target, inner_lrs),
                    pde,
                    fenics_functions,
                    gt_params,
                    get_final_model,
                    maml_def,
                    FLAGS.inner_steps,
                )

            if FLAGS.expt_name is not None:
                plt.savefig(os.path.join(path, "viz_step_{}.png".format(step)), dpi=800)

            if tflogger is not None:
                tflogger.log_plots("Ground truth comparison", [plt.gcf()], step)

            # save model
            optimizer_target = flax.serialization.to_state_dict(optimizer.target)
            inner_lrs_target = flax.serialization.to_state_dict(inner_lr_get(inner_lr_state))
            with open(os.path.join(path, "model_step_{}.pickle".format(step)), "wb") as f:
                pickle.dump([optimizer_target, inner_lrs_target], f, protocol=pickle.HIGHEST_PROTOCOL)

    # if FLAGS.expt_name is not None:
    #     outfile.close()

    plt.figure()
    if FLAGS.pde == 'td_burgers':
        trainer_util.plot_model_time_series_new(
            (optimizer_target, inner_lrs),
            pde,
            fenics_functions,
            gt_params,
            get_final_model,
            maml_def,
            FLAGS.inner_steps,
        )
    else:
        trainer_util.compare_plots_with_ground_truth(
            (optimizer_target, inner_lrs),
            pde,
            fenics_functions,
            gt_params,
            get_final_model,
            maml_def,
            FLAGS.inner_steps,
        )
    if FLAGS.expt_name is not None:
        plt.savefig(os.path.join(path, "viz_final.png"), dpi=800)
    else:
        plt.show()


if __name__ == "__main__":
    app.run(main)
