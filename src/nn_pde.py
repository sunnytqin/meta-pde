"""Fit NN to one PDE."""

from jax.config import config
from .util import trainer_util
import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from jax.experimental import optimizers
from adahessianJax import grad_and_hessian

from .nets import maml
from .get_pde import get_pde

from functools import partial
import flax
from flax import nn
import fenics as fa

from .util import pcgrad
from .util.timer import Timer

from .util import jax_tools


import time

import matplotlib.pyplot as plt
from collections import deque
import pdb
import sys
import os
import logging
import re
import pickle

from .util import common_flags

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("bsize", 1, "batch size (in tasks)")
flags.DEFINE_float("outer_lr", 1e-5, "outer learning rate")


def main(argv):
    if FLAGS.out_dir is None:
        FLAGS.out_dir = FLAGS.pde + "_nn_results"

    FLAGS.n_eval = 1
    FLAGS.fixed_num_pdes = 1

    pde = get_pde(FLAGS.pde)

    path, log, tflogger = trainer_util.prepare_logging(FLAGS.out_dir, FLAGS.expt_name)

    log(FLAGS.flags_into_string())

    # --------------------- Defining the training algorithm --------------------
    def loss_fn(field_fn, points, params):
        boundary_losses, domain_losses = pde.loss_fn(field_fn, points, params)

        loss = FLAGS.bc_weight * np.sum(
            np.array([bl for bl in boundary_losses.values()])
        ) + np.sum(np.array([dl for dl in domain_losses.values()]))

        return loss, {**boundary_losses, **domain_losses}

    def task_loss_fn(key, model):
        # The input key is terminal
        k1, k2 = jax.random.split(key, 2)
        params = pde.sample_params(k1)
        points = pde.sample_points(k2, FLAGS.outer_points, params)
        return loss_fn(model, points, params)

    @jax.jit
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
            losses_and_grad_norms[k] = (loss_val, loss_grad_norm)
        return losses_and_grad_norms

    @jax.jit
    def batch_loss_fn(key, model):
        vmap_task_loss_fn = jax.vmap(task_loss_fn, (0, None))

        keys = jax.random.split(key, FLAGS.bsize)
        loss, loss_dict = vmap_task_loss_fn(keys, model)
        #loss, loss_dict, params_new = task_loss_fn(key, model, fa_p)

        loss_aux = {}  # store original loss by loss type

        # get original gradients
        for k in loss_dict:
            loss_aux[k] = np.mean(loss_dict[k])

        return np.sum(loss), loss_aux

    # --------------------- Defining the evaluation functions --------------------
    def get_final_model(unused_key, model,
                         _unused_params=None,
                         _unused_num_steps=None,
                         _unused_meta_alg_def=None):
        # Input key is terminal
        return model

    def make_coef_func(key, model, params, coords):
        # Input key is terminal
        final_model = get_final_model(key, model, params)

        return np.squeeze(final_model(coords))

    @jax.jit
    def train_step(key, optimizer):
        (loss, loss_aux), batch_grad = jax.value_and_grad(
            batch_loss_fn, argnums=1, has_aux=True
        )(key, optimizer.target)

        grad_norm = np.sqrt(
            jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_util.tree_map(lambda x: np.sum(x ** 2), batch_grad),
            )
        )

        batch_grad = jax.lax.cond(
            grad_norm > FLAGS.grad_clip,
            lambda grad_tree: jax.tree_util.tree_map(
                lambda x: FLAGS.grad_clip * x / grad_norm, grad_tree
            ),
            lambda grad_tree: grad_tree,
            batch_grad
        )

        if FLAGS.optimizer == "adahessian":
            optimizer = optimizer.apply_gradient(batch_grad, batch_hess)
        else:
            optimizer = optimizer.apply_gradient(batch_grad)

        return optimizer, loss, loss_aux, grad_norm

    @jax.jit
    def validation_losses(model):
        return task_loss_fn(jax.random.PRNGKey(0), model)[0]

    # ----- initialize model  ----
    Field = pde.BaseField.partial(
        sizes=[FLAGS.layer_size for _ in range(FLAGS.num_layers)],
        dense_args=(),
        nonlinearity=np.sin if FLAGS.siren else nn.swish,
        omega=FLAGS.siren_omega,
        omega0=FLAGS.siren_omega0,
        log_scale=FLAGS.log_scale,
        #use_laaf=FLAGS.laaf,
        #use_nlaaf=FLAGS.nlaaf,
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

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

        init_params = flax.serialization.from_state_dict(init_params, optimizer_target['params'])

        optimizer = trainer_util.get_optimizer(Field, init_params)

    log('NN model:', jax.tree_map(lambda x: x.shape, init_params))

    # ----- sample param and get fenics ground truth -----
    key, gt_key, gt_points_key = jax.random.split(key, 3)

    gt_keys = jax.random.split(gt_key, FLAGS.n_eval)
    gt_params = vmap(pde.sample_params)(gt_keys)
    print("gt_params: {}".format(gt_params))

    fenics_functions, fenics_vals, coords = trainer_util.get_ground_truth_points(
        pde, jax_tools.tree_unstack(gt_params), gt_points_key
    )

    if FLAGS.pde == 'td_burgers':
        t_list = fenics_functions[0].timesteps_list
    else:
        t_list = np.array([0])
        tile_idx = coords.shape[1]

    # ----- train NN -----
    time_last_log = time.time()
    try:
        print(optimizer.target.params['Dense_0'])
    except:
        print(optimizer.target.params['0']['Dense_0'])

    for step in range(FLAGS.outer_steps + 1):
        key, subkey = jax.random.split(key)
        with Timer() as t:
            optimizer, loss, loss_aux, grad_norm = train_step(subkey, optimizer)

        # ---- This big section is logging a bunch of debug stats
        # loss grad norms; plotting the sampled points; plotting the vals at those
        # points; plotting the losses at those points.

        if (
            FLAGS.measure_grad_norm_every > 0
            and step % FLAGS.measure_grad_norm_every == 0
        ):
            loss_vals_and_grad_norms = get_grad_norms(subkey, optimizer.target)
            loss_vals_and_grad_norms = {k: (float(v[0]), float(v[1]))
                                        for k, v in loss_vals_and_grad_norms.items()}
            log("loss vals and grad norms: ", loss_vals_and_grad_norms)

            if tflogger is not None:
                for k in loss_vals_and_grad_norms:
                    tflogger.log_scalar(
                        "grad_norm_{}".format(k),
                        float(loss_vals_and_grad_norms[k][1]),
                        step,
                    )
                #_k1, _k2 = jax.random.split(
                #    jax.random.split(subkey, FLAGS.bsize)[0], 2
                #)
                #_params = jax_tools.tree_unstack(gt_params)[0]
                #_points = pde.sample_points(_k2, FLAGS.outer_points, _params)
                #plt.figure()
                #for _pointsi in _points:
                #    plt.scatter(_pointsi[:, 0], _pointsi[:, 1], label=f'n_points={_pointsi.shape[0]}')
                #    plt.legend()
                #tflogger.log_plots("Points", [plt.gcf()], step)
                #_all_points = np.concatenate(_points)
                #_vals = optimizer.target(_all_points)
                #_vals = _vals.reshape((_vals.shape[0], -1))
                #_boundary_losses, _domain_losses = jax.vmap(
                #    lambda x: pde.loss_fn(
                #        optimizer.target,
                #        (x.reshape(1, -1) for _ in range(len(_points))),
                #        _params,
                #    )
                #)(_all_points)

                #_, _domain_losses_domain = jax.vmap(
                #    lambda x: pde.loss_fn(
                #        optimizer.target,
                #        (x.reshape(1, -1) for _ in range(len(_points))),
                #        _params,
                #    )
                #)(np.squeeze(coords))

                #_all_losses = {**_boundary_losses, **_domain_losses}
                #for _losskey in _all_losses:
                #    plt.figure()
                #    _loss = _all_losses[_losskey]
                #    while len(_loss.shape) > 1:
                #        _loss = _loss.mean(axis=1)
                #    clrs = plt.scatter(
                #        _all_points[:, 0],
                #        _all_points[:, 1],
                #        c=_loss[: len(_all_points)],
                #    )
                #    plt.colorbar(clrs)
                #    tflogger.log_plots("{}".format(_losskey), [plt.gcf()], step)

                #for dim in range(_vals.shape[1]):
                #    plt.figure()
                #    clrs = plt.scatter(
                #        _all_points[:, 0], _all_points[:, 1], c=_vals[:, dim]
                #    )
                #    plt.colorbar(clrs)
                #    tflogger.log_plots(
                #        "Outputs dim {}".format(dim), [plt.gcf()], step
                #    )
                #_outputs_on_coords = optimizer.target(coords[0])
                #if len(_outputs_on_coords.shape) == 1:
                #    _outputs_on_coords = _outputs_on_coords[:, None]

                #for dim in range(_vals.shape[1]):
                #    plt.figure()
                #    clrs = plt.scatter(
                #        coords[0][:, 0], coords[0][:, 1], c=_outputs_on_coords[:, dim]
                #    )
                #    plt.colorbar(clrs)
                #    tflogger.log_plots(
                #        "NN_on_coords dim {}".format(dim), [plt.gcf()], step
                #    )

                #    plt.figure()
                #    clrs = plt.scatter(
                #        coords[0][:, 0], coords[0][:, 1], c=fenics_vals[0][:, dim]
                #    )
                #    plt.colorbar(clrs)
                #    tflogger.log_plots(
                #        "Fenics_on_coords dim {}".format(dim), [plt.gcf()], step
                #    )

                #    norm_to_plot = np.linalg.norm(fenics_vals[0], axis=1)
                    
                #    plt.figure(figsize=(24, 8))
                #    for i, t_plot in enumerate(t_list):
                #        plt.subplot(2, int(np.ceil(len(t_list)/2)), i + 1)
                #        t_idx = np.squeeze(np.arange(i * tile_idx, (i + 1) * tile_idx))
                #        clrs = plt.scatter(
                #        coords[0][t_idx, 0], coords[0][t_idx, 1], c=norm_to_plot[t_idx]
                #        )
                #        plt.title(f"t = {t_plot:.3f}")
                #        plt.colorbar(clrs)
                #    tflogger.log_plots(
                #        "Ground_truth_norm_on_coords dim {}".format(dim), [plt.gcf()], step
                #    )

                #    mse_to_plot = (fenics_vals[0][:, dim] - _outputs_on_coords[:, dim])**2

                #    plt.figure(figsize=(24, 8))
                #    for i, t_plot in enumerate(t_list):
                #        plt.subplot(2, int(np.ceil(len(t_list)/2)), i + 1)
                #        t_idx = np.squeeze(np.arange(i * tile_idx, (i + 1) * tile_idx))
                #        clrs = plt.scatter(
                #        coords[0][t_idx, 0], coords[0][t_idx, 1], c=mse_to_plot[t_idx]
                #        )
                #        plt.title(f"t = {t_plot:.3f}")
                #        plt.colorbar(clrs)
                #    tflogger.log_plots(
                #        "Residual_on_coords dim {}".format(dim), [plt.gcf()], step
                #    )

                #_domain_losses_domain = {**_domain_losses_domain}
                #for _losskey in _domain_losses_domain:
                #    _loss = _domain_losses_domain[_losskey]
                #    while len(_loss.shape) > 1:
                #        _loss = _loss.mean(axis=1)

                #    tmp_filenames = []
                #    for i, domain_t in enumerate(t_list):
                #        plt.figure(figsize=(20, 8))
                #        plt.subplot(1, 2, 1)
                #        _coords_index = np.squeeze(np.arange(i * tile_idx, (i + 1) * tile_idx))
                #        #_coords_index = np.isclose(np.squeeze(coords)[:, 2], domain_t)
                #        clrs = plt.scatter(
                #            coords[0][_coords_index, 0], coords[0][_coords_index, 1],
                #            c=np.log(_loss[_coords_index]),
                #        )
                #        plt.colorbar(clrs)
                #        plt.title(f"t = Validation Loss {domain_t:.3f}")

                #        mse_to_plot = np.linalg.norm((fenics_vals[0] - _outputs_on_coords), axis=1)
                #        plt.subplot(1, 2, 2)
                #        clrs = plt.scatter(
                #            coords[0][_coords_index, 0], coords[0][_coords_index, 1],
                #            c=mse_to_plot[_coords_index]
                #        )
                #        plt.colorbar(clrs)
                #        plt.title(f"t = Residual {domain_t:.3f}")

                #        if FLAGS.pde == 'td_burgers':
                #            plt.savefig(f"td_burger_errors_{i}.png")
                #            tmp_filenames.append(f"td_burger_errors_{i}.png")
                #        else:
                #            fig_out = os.path.join(path, f"{FLAGS.pde}_errors_step_{step}.png")
                #            plt.savefig(fig_out)
                #        # tflogger.log_plots("Validation Loss in {}".format(domain_t), [plt.gcf()], step)
                #    if FLAGS.pde == 'td_burgers':
                #        gif_out = os.path.join(path, "td_burger_errors_step_{}.gif".format(step))
                #        pde.build_gif(tmp_filenames, outfile=gif_out)

        if step % FLAGS.log_every == 0:
            with Timer() as deploy_timer:
                mse, norms, rel_err, per_dim_rel_err, rel_err_std, t_rel_sq_err = trainer_util.vmap_validation_error(
                    optimizer.target, gt_params, coords, fenics_vals, make_coef_func
                )
                mse.block_until_ready()
            deployment_time = deploy_timer.interval / FLAGS.n_eval

            val_loss = validation_losses(optimizer.target)

            if step > 0:
                log("total time {} steps: {}".format(FLAGS.log_every,
                                               time.time() - time_last_log))
            time_last_log = time.time()

            log(
                "step: {}, loss: {}, val_loss: {}, val_mse: {}, "
                "val_rel_err: {}, val_rel_err_std: {}, val_true_norms: {}, "
                "per_dim_rel_err: {}, deployment_time: {}, grad_norm: {}, "
                "per_time_step_error: {} ,per train step time: {}".format(
                    step,
                    loss,
                    val_loss,
                    mse,
                    rel_err,
                    rel_err_std,
                    norms,
                    per_dim_rel_err,
                    deployment_time,
                    grad_norm,
                    t_rel_sq_err,
                    t.interval,
                )
            )

            if tflogger is not None:
                # A lot of these names have unnecessary "meta_"
                # just for consistency with maml_pde and leap_pde
                tflogger.log_scalar("meta_loss", float(np.mean(loss)), step)
                tflogger.log_scalar("val_loss", float(np.mean(val_loss)), step)

                tflogger.log_scalar("val_rel_mse", float(rel_err), step)
                tflogger.log_scalar("std_val_rel_mse", float(rel_err_std), step)

                tflogger.log_scalar("val_mse", float(mse), step)

                if FLAGS.pde == 'td_burgers':
                    plt.figure()
                    plt.plot(t_list, t_rel_sq_err, '.')
                    plt.xlabel('t')
                    plt.ylabel('val rel err')
                    plt.yscale('log')
                    tflogger.log_plots(
                        "Per time step relative error", [plt.gcf()], step
                    )

                for i in range(len(per_dim_rel_err)):
                    tflogger.log_scalar(
                        "val_rel_error_dim_{}".format(i), float(per_dim_rel_err[i]), step
                    )
                    tflogger.log_scalar("val_norm_dim_{}".format(i), float(norms[i]), step)

                tflogger.log_scalar("meta_grad_norm", float(grad_norm), step)
                tflogger.log_scalar("step_time", t.interval, step)
                for k in loss_aux:
                    tflogger.log_scalar("meta_" + k, float(np.mean(loss_aux[k])), step)

                if step % FLAGS.viz_every == 0:
                    # These take lots of filesize so only do them sometimes

                    for k, v in jax_tools.dict_flatten(optimizer.target.params):
                        tflogger.log_histogram("Param: " + k, v.flatten(), step)
                        if 'scale' in k:
                            print("Scale params: {}: {}".format(k, v))

            log("time for logging {}".format(time.time() - time_last_log))
            time_last_log = time.time()

        if FLAGS.viz_every > 0 and step % FLAGS.viz_every == 0:
            if FLAGS.pde == 'td_burgers':
                plt.figure()
                plot_model_time_series = trainer_util.plot_model_time_series_new

                plot_model_time_series(
                    optimizer.target,
                    pde,
                    fenics_functions,
                    gt_params,
                    get_final_model,
                    None,
                    0,
                )
                if FLAGS.expt_name is not None:
                    plt.savefig(os.path.join(path, "viz_step_{}.png".format(step)), dpi=800)
                #gif_out = os.path.join(path, "td_burger_step_{}.gif".format(step))
                #pde.build_gif(tmp_filenames, outfile=gif_out)

            else:
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

                if FLAGS.expt_name is not None:
                    plt.savefig(os.path.join(path, "viz_step_{}.png".format(step)), dpi=800)

                #if tflogger is not None:
                    #tflogger.log_plots("Ground truth comparison", [plt.gcf()], step)

            # save model
            optimizer_target = flax.serialization.to_state_dict(optimizer.target)
            with open(os.path.join(path, "model_step_{}.pickle".format(step)), "wb") as f:
                pickle.dump(optimizer_target, f, protocol=pickle.HIGHEST_PROTOCOL)


    #if FLAGS.expt_name is not None:
    #    outfile.close()

if __name__ == "__main__":
    app.run(main)




