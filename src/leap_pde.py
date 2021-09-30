"""Use LEAP to amortize fitting a NN across a class of PDEs."""
from jax.config import config

import jax
import jax.numpy as np
import numpy as npo
import re
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
import pickle
import shutil
from copy import deepcopy
from collections import namedtuple

import time


from .util import common_flags

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("bsize", 16, "batch size (in tasks)")
flags.DEFINE_float("outer_lr", 1e-3, "outer learning rate")

flags.DEFINE_float("inner_lr", 3e-5, "inner learning rate")
flags.DEFINE_float("inner_grad_clip", 1e14, "inner grad clipping")

flags.DEFINE_integer("inner_steps", 10, "num_inner_steps")


def main(argv):
    if FLAGS.out_dir is None:
        FLAGS.out_dir = FLAGS.pde + "_leap_results"

    if FLAGS.load_model_from_expt is not None:
        FLAGS.out_dir = os.path.join(FLAGS.out_dir, 'rerun')

    pde = get_pde(FLAGS.pde)

    path, log, tflogger = trainer_util.prepare_logging(FLAGS.out_dir, FLAGS.expt_name)

    log(FLAGS.flags_into_string())

    with open(os.path.join(path, "flags_config.txt"), "w") as f:
        f.write(FLAGS.flags_into_string())

    # --------------------- Defining the meta-training algorithm --------------------

    def loss_fn(field_fn, points, params):
        boundary_losses, domain_losses = pde.loss_fn(field_fn, points, params)

        loss = FLAGS.bc_weight * np.sum(
            np.array([bl for bl in boundary_losses.values()])
        ) + np.sum(np.array([dl for dl in domain_losses.values()]))

        if FLAGS.laaf:
            assert not FLAGS.nlaaf
            laaf_loss = FLAGS.laaf_weight * trainer_util.loss_laaf(field_fn)
            laaf_loss_dict = {'laaf_loss': laaf_loss}

            # recompute loss
            loss = loss + laaf_loss

        elif FLAGS.nlaaf:
            assert not FLAGS.laaf
            laaf_loss = FLAGS.laaf_weight * trainer_util.loss_nlaaf(field_fn)
            laaf_loss_dict = {'laaf_loss': laaf_loss}

            # recompute loss
            loss = loss + laaf_loss

        # return the total loss, and as aux a dict of individual losses
        if FLAGS.laaf or FLAGS.nlaaf:
            return loss, {**boundary_losses, **domain_losses, **laaf_loss_dict}
        else:
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
        omega=FLAGS.siren_omega,
        omega0=FLAGS.siren_omega0,
        log_scale=FLAGS.log_scale,
        use_laaf=FLAGS.laaf,
        use_nlaaf=FLAGS.nlaaf,
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    if FLAGS.pde == 'td_burgers':
        _, init_params = Field.init_by_shape(subkey, [((1, 3), np.float32)])
    else:
        _, init_params = Field.init_by_shape(subkey, [((1, 2), np.float32)])

    if FLAGS.load_model_from_expt is not None:
        model_dir = FLAGS.pde + "_leap_results"
        model_path = os.path.join(model_dir, FLAGS.load_model_from_expt)
        model_file = npo.array(
            [f for f in os.listdir(model_path) if "leap_step" in f]
        )
        steps = npo.zeros_like(model_file, dtype=int)
        for i, f in enumerate(model_file):
            steps[i] = re.findall('[0-9]+', f)[-1]
        model_file = model_file[
            np.argsort(steps)[-1]
        ]
        log('load pre-trained model from file: ', model_file)
        with open(os.path.join(model_path, model_file), 'r') as f:
            optimizer_target_prev = f.read()
        init_params = flax.serialization.from_state_dict(init_params, optimizer_target_prev)

    log(
        'NN model:', jax.tree_map(lambda x: x.shape, init_params)
    )

    #for k, v in init_params.items():
    #    if type(v) is not dict:
    #        print(f"-> {k}: {v.shape}")
    #    else:
    #        print(f"-> {k}")
    #        for k2, v2 in v.items():
    #            if type(v2) is not dict:
    #                print(f"  -> {k2}: {v2.shape}")
    #            else:
    #                print(f"  -> {k2}")
    #                for k3, v3 in v2.items():
    #                    print(f"    -> {k3}: {v3.shape}")

    optimizer = trainer_util.get_optimizer(Field, init_params)

    # --------------------- Defining the evaluation functions --------------------

    @partial(jax.jit, static_argnums=(3, 4))
    def get_final_model(key, model, params, inner_steps, leap_def):
        # Input key is terminal
        k1, k2 = jax.random.split(key, 2)
        inner_points = pde.sample_points(k1, FLAGS.inner_points, params)
        inner_loss_fn = lambda key, field_fn: loss_fn(field_fn, inner_points, params)

        temp_leap_def = leap_def._replace(inner_steps=inner_steps)
        final_model = jax.lax.cond(
            inner_steps != 0,
            lambda _: leap.single_task_rollout(
                temp_leap_def, k2, model, inner_loss_fn
            )[0],
            lambda _: model,
            0,
        )
        return final_model

    @partial(jax.jit, static_argnums=(4, 5))
    def make_coef_func(key, model_and_lrs, params, coords, inner_steps, leap_def):
        # Input key is terminal
        final_model = get_final_model(key, model_and_lrs, params, inner_steps, leap_def)

        return np.squeeze(final_model(coords))

    partial_make_coef_func = lambda key, model, params, coords: make_coef_func(
            key, model, params, coords, leap_def.inner_steps, leap_def)

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
                jax.tree_util.tree_map(lambda x: np.sum(x ** 2), meta_grad),
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

    if FLAGS.pde == 'td_burgers':
        t_list = []
        for i in range(FLAGS.num_tsteps):
            tile_idx = coords.shape[1] // FLAGS.num_tsteps
            t_idx = np.squeeze(np.arange(i * tile_idx, (i + 1) * tile_idx))
            t_unique = np.unique(coords[:, t_idx, 2])
            t_list.append(np.squeeze(t_unique))
            assert len(t_unique) == 1

    time_last_log = time.time()

    # --------------------- Run LEAP --------------------

    for step in range(FLAGS.outer_steps):
        key, subkey = jax.random.split(key, 2)

        with Timer() as t:
            optimizer, losses, meta_grad_norm = train_step(subkey, optimizer)

        if np.isnan(np.mean(losses[0][:, -1])):
            log("encountered nan at at step {}".format(step))
            # save final model
            #with open(os.path.join(path, "leap_step_final.txt".format(step)), "w") as f:
            #    f.write(optimizer_target_prev)
            break

        if step % FLAGS.log_every == 0:
            with Timer() as deploy_timer:
                mse, norms, rel_err, per_dim_rel_err, rel_err_std, t_rel_sq_err = trainer_util.vmap_validation_error(
                    optimizer.target, gt_params, coords,
                    fenics_vals,
                    partial_make_coef_func)
                mse.block_until_ready()
            deployment_time = deploy_timer.interval / FLAGS.n_eval

            val_losses = validation_losses(optimizer.target)

            #optimizer_target_prev = flax.serialization.to_state_dict(optimizer.target)

            log(
                "step: {}, meta_loss: {}, val_meta_loss: {}, val_mse: {}, "
                "val_rel_err: {}, val_rel_err_std: {}, val_true_norms: {}, "
                "per_dim_rel_err: {}, per_time_step_error: {}, deployment_time: {}"
                "meta_grad_norm: {}, time: {}, key: {}, subkey: {}".format(
                    step,
                    np.mean(losses[0][:, -1]),
                    np.mean(val_losses[0][:, -1]),
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
                tflogger.log_scalar("val_rel_mse", float(rel_err), step)
                tflogger.log_scalar("std_val_rel_mse", float(rel_err_std), step)

                tflogger.log_scalar("val_mse", float(mse), step)

                for i in range(len(per_dim_rel_err)):
                    tflogger.log_scalar(
                        "val_rel_error_dim_{}".format(i), float(per_dim_rel_err[i]), step
                    )
                    tflogger.log_scalar("val_norm_dim_{}".format(i), float(norms[i]), step)

                for k in losses[1]:
                    tflogger.log_scalar(
                        "meta_" + k, float(np.mean(losses[1][k][:, -1])), step
                    )

                tflogger.log_scalar("meta_grad_norm", float(meta_grad_norm), step)
                tflogger.log_scalar("step_time", t.interval, step)

                if step % FLAGS.viz_every == 0:

                    if FLAGS.pde == 'td_burgers':
                        plt.figure()
                        plt.plot(t_list, t_rel_sq_err, '.')
                        plt.xlabel('t')
                        plt.ylabel('val rel err')
                        tflogger.log_plots(
                            "Per time step relative error", [plt.gcf()], step
                        )

                    plt.figure()
                    plt.plot(np.arange(FLAGS.inner_steps + 1), np.mean(losses[0], axis=0))
                    plt.xlabel('Inner Step')
                    plt.ylabel('Loss')
                    plt.yscale('log')
                    tflogger.log_plots(
                        "Per inner step step loss", [plt.gcf()], step
                    )

                    plt.figure()
                    plt.plot(np.arange(FLAGS.inner_steps + 1), np.mean(val_losses[0], axis=0))
                    plt.xlabel('Inner Step')
                    plt.ylabel('Val Loss')
                    plt.yscale('log')
                    tflogger.log_plots(
                        "Per inner step step val loss", [plt.gcf()], step
                    )

                for inner_step in range(FLAGS.inner_steps + 1):
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
                    #for k in losses[1]:
                    #    tflogger.log_scalar(
                    #        "{}_step_{}".format(k, inner_step),
                    #        float(np.mean(losses[1][k][:, inner_step])),
                    #        step,
                    #    )

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

            if FLAGS.pde == 'td_burgers':
                tmp_filenames = trainer_util.plot_model_time_series(
                    optimizer.target,
                    pde,
                    fenics_functions,
                    gt_params,
                    get_final_model,
                    leap_def,
                    FLAGS.inner_steps,
                )
                gif_out = os.path.join(path, "td_burger_step_{}.gif".format(step))
                pde.build_gif(tmp_filenames, outfile=gif_out)

            # save model
            optimizer_target = flax.serialization.to_state_dict(optimizer.target)
            with open(os.path.join(path, "leap_step_{}.pickle".format(step)), "wb") as f:
                pickle.dump(optimizer_target, f, protocol=pickle.HIGHEST_PROTOCOL)

    #if FLAGS.expt_name is not None:
    #    outfile.close()

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

    if FLAGS.pde == 'td_burgers':
        tmp_filenames = trainer_util.plot_model_time_series(
            optimizer.target,
            pde,
            fenics_functions,
            gt_params,
            get_final_model,
            leap_def,
            FLAGS.inner_steps,
        )
        gif_out = os.path.join(path, "td_burger_final.gif".format(step))
        pde.build_gif(tmp_filenames, outfile=gif_out)


if __name__ == "__main__":
    app.run(main)
