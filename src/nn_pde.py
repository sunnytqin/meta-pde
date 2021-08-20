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
    FLAGS.vary_bc = False
    FLAGS.vary_source = False
    FLAGS.vary_geometry = False

    pde = get_pde(FLAGS.pde)

    path, log, tflogger = trainer_util.prepare_logging(FLAGS.out_dir, FLAGS.expt_name)

    log(FLAGS.flags_into_string())

    # --------------------- Defining the training algorithm --------------------

    def loss_fn(field_fn, points, params):
        boundary_losses, domain_losses = pde.loss_fn(field_fn, points, params)
        loss = np.sum(
            np.array([bl for bl in boundary_losses.values()])
        ) + np.sum(
            np.array([dl for dl in domain_losses.values()]))

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


    def task_loss_fn(key, model):
        # The input key is terminal
        k1, k2 = jax.random.split(key, 2)
        params = pde.sample_params(k1)
        points = pde.sample_points(k2, FLAGS.outer_points, params)
        return loss_fn(model, points, params)



    @jax.jit
    def get_grad_norms(key, model):
        _, (loss_dict, _) = batch_loss_fn(key, model, None)
        losses_and_grad_norms = {}
        for k in loss_dict:
            single_loss_fn = lambda model: batch_loss_fn(key, model, None)[1][0][k]
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
    def perform_annealing(loss_grads, bc_weights_prev):
        bc_weights = {FLAGS.domain_loss: 1.}
        for k, loss_grad in loss_grads.items():
            if k == FLAGS.domain_loss:
                if FLAGS.annealing_l2:
                    domain_loss_grad_max = np.sum(
                        np.array(jax.tree_flatten(
                        jax.tree_util.tree_map(lambda x: np.sum(x**2), loss_grad))[0]
                        )
                    )
                else:
                    domain_loss_grad_max = np.max(
                        np.array(jax.tree_flatten(
                        jax.tree_util.tree_map(lambda x: np.sum(np.abs(x)), loss_grad))[0]
                        )
                    )
            else:
                if FLAGS.annealing_l2:
                    loss_grad_mean = np.sum(
                        np.array(jax.tree_flatten(
                            jax.tree_util.tree_map(lambda x: np.sum(x**2), loss_grad))[0]
                        )
                    )
                else:
                    loss_grad_mean = np.sum(
                        np.array(jax.tree_flatten(
                            jax.tree_util.tree_map(lambda x: np.sum(np.abs(x)), loss_grad))[0]
                        )
                    )
                bc_weights[k] = loss_grad_mean
        for k in loss_grads:
            if k == FLAGS.domain_loss:
                continue
            if bc_weights_prev is not None:
                bc_weights[k] = (1 - FLAGS.annealing_alpha) * bc_weights_prev[k] + \
                                FLAGS.annealing_alpha * (domain_loss_grad_max / bc_weights[k])
            else:
                bc_weights[k] = domain_loss_grad_max / bc_weights[k]
        return bc_weights

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

    #for k, v in init_params.items():
    #    print(f"Layer {k}")
    #    if type(v) is not dict:
    #        print(f"     info -> {v.shape}")
    #    else:
    #        for k2, v2 in v.items():
    #            print(f"     info -> {k2}: {v2.shape}")

    for k, v in init_params.items():
        print(f"{k}")
        if type(v) is not dict:
            print(f"   -> {k}: {v.shape}")
        else:
            print(f"   -> {k}")
            for k2, v2 in v.items():
                if type(v2) is not dict:
                    print(f"     -> {k2}: {v2.shape}")
                else:
                    print(f"     -> {k2}")
                    for k3, v3 in v2.items():
                        print(f"      i -> {k3}: {v3.shape}")

    optimizer = trainer_util.get_optimizer(Field, init_params)

    dummy_flat_grad, dummy_treedef = jax.tree_flatten(optimizer.target)
    dummy_shapes = tuple(x.shape for x in dummy_flat_grad)
    dummy_sizes = tuple(x.size for x in dummy_flat_grad)
    fins = np.cumsum(np.array(dummy_sizes))
    starts = fins - np.array(dummy_sizes)

    fins = tuple(fins.tolist())
    starts = tuple(starts.tolist())

    @jax.jit
    def reform(flat_arr):
        out_list = []
        for i in range(len(dummy_shapes)):
            out_list.append(np.reshape(flat_arr[starts[i]:fins[i]],
                                       dummy_shapes[i]))
        return jax.tree_unflatten(dummy_treedef, out_list)

    @jax.jit
    def perform_pcgrad(loss_grads):
        # define projection function
        project = partial(pcgrad.project_grads, FLAGS.pcgrad_norm)

        loss_grads_flat = {k: np.concatenate([arr.flatten() for arr in jax.tree_flatten(v)[0]])
            for k, v in loss_grads.items()}
        # for new individual loss gradient
        loss_grads_new = {}

        total_grad_new = 0.

        # perform PC grad for each loss type
        for k in loss_grads_flat:
            grad = loss_grads_flat[k]
            other_grads = {k1: v1 for k1, v1 in loss_grads_flat.items() if k1 != k}
            loss_grad_new = project(grad, *other_grads.values())
            # update the new projected individual loss function
            loss_grads_new[k] = reform(loss_grad_new)

            # update gradient for the entire loss function
            total_grad_new = loss_grad_new + total_grad_new

        total_grad_new = reform(total_grad_new)

        return total_grad_new, loss_grads_new

    @partial(jax.jit, static_argnums=[2])
    def batch_loss_fn(key, model, bc_weights_prev):
        vmap_task_loss_fn = jax.vmap(task_loss_fn, (0, None))

        keys = jax.random.split(key, FLAGS.bsize)
        _, loss_dict = vmap_task_loss_fn(keys, model)
        #loss, loss_dict, params_new = task_loss_fn(key, model, fa_p)

        loss_aux = {}  # store original loss by loss type
        loss_grads = {}  # store original loss gradient by loss type

        # get original gradients
        for k in loss_dict:
            loss_aux[k] = np.mean(loss_dict[k])
            # do nothing if not annealing and not pcgrad-ing
            if (not FLAGS.annealing and not FLAGS.pcgrad):
                continue
            #assert False
            single_loss_fn = lambda model: np.mean(vmap_task_loss_fn(keys, model)[1][k])
            _, loss_grad = jax.value_and_grad(single_loss_fn)(model)
            loss_grads[k] = loss_grad

        # perform PC grad
        if FLAGS.pcgrad:
            assert False
            # place holder for new total loss gradient
            #total_grad_new = [np.zeros_like(x)
            #                  for x in jax.tree_flatten(loss_grad)[0]]
            # pdb.set_trace()
            _, loss_grads = perform_pcgrad(loss_grads)

        # perform weight annealing
        if FLAGS.annealing:
            bc_weights = perform_annealing(loss_grads, bc_weights_prev)
            loss = np.sum(np.array([bc_weights[k] * v for k, v in loss_aux.items()]))
        else:
            bc_weights = None
            loss = np.sum(np.array([v*(FLAGS.bc_weight if k not in FLAGS.domain_loss else 1.) for k, v in loss_aux.items()]))


        return loss, (loss_aux, bc_weights)

    #bc_weights = None

    # --------------------- Defining the evaluation functions --------------------


    def get_final_model(unused_key, model,
                         _unused_params=None,
                         _unused_num_steps=None,
                         _unused_meta_alg_def=None):
        # Input key is terminal
        return model

    def get_final_model_old(unused_key, model,
                         _unused_params=None,
                         _unused_num_steps=None,
                         _unused_meta_alg_def=None
                         ):
        """deprecated"""
        if fa_p is not None:
            def final_model(x):
                velocity = model(x)
                pressure = fa_p(x)
                assembled = np.stack((velocity[:, 0], velocity[:, 1], pressure), axis=-1)
                return assembled
            return final_model
        if FLAGS.pde == 'td_burgers':
            def final_model(x):
                if x.shape[-1] == 2:  # add time axis if missing time-dimension
                    t_list = np.linspace(FLAGS.tmin, FLAGS.tmax, FLAGS.num_tsteps, endpoint=False)
                    t_idx = npo.unique(npo.linspace(0, len(t_list), 2, endpoint=False, dtype=int)[1:])
                    t_supp = np.squeeze(t_list[t_idx])
                    print(f'plotting final model at t = {t_supp}')
                    t_supp = np.repeat(t_supp, x.shape[0]).reshape(x.shape[0], 1)

                    x = np.concatenate([x, t_supp], axis=1)
                return model(x)

            return final_model
        else:
            return model


    def make_coef_func(key, model, params, coords):
        # Input key is terminal
        final_model = get_final_model(key, model, params)

        return np.squeeze(final_model(coords))



    @jax.jit
    def train_step(key, optimizer, bc_weights_prev):

        if FLAGS.optimizer == "adahessian":
            k1, k2 = jax.random.split(key)
            loss, (loss_aux, bc_weights) = batch_loss_fn(
                k1, optimizer.target, bc_weights_prev)
            batch_grad, batch_hess = grad_and_hessian(
                lambda model: batch_loss_fn(k1, model, bc_weights)[0],
                (optimizer.target,),
                k2,
            )
        else:
            (loss, (loss_aux, bc_weights)), batch_grad = jax.value_and_grad(
                batch_loss_fn, argnums=1, has_aux=True
            )(key, optimizer.target, bc_weights_prev)


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

        return optimizer, loss, loss_aux, grad_norm, bc_weights

    @jax.jit
    def validation_losses(model):
        return task_loss_fn(jax.random.PRNGKey(0), model)[0]

    key, gt_key, gt_points_key = jax.random.split(key, 3)
    #print('sample param key: ', gt_key)

    gt_keys = jax.random.split(gt_key, FLAGS.n_eval)
    gt_params = vmap(pde.sample_params)(gt_keys)
    print("gt_params: {}".format(gt_params))

    fenics_functions, fenics_vals, coords = trainer_util.get_ground_truth_points(
        pde, jax_tools.tree_unstack(gt_params), gt_points_key
    )

    if FLAGS.pde == 'td_burgers':
        FLAGS.tmax_nn = 1e-4
        early_stopping_tracker = deque()
        propagate_time = False
        last_prop_step = 0

        t_list = []
        for i in range(FLAGS.num_tsteps):
            tile_idx = coords.shape[1] // FLAGS.num_tsteps
            t_idx = np.squeeze(np.arange(i * tile_idx, (i + 1) * tile_idx))
            t_unique = np.unique(coords[:, t_idx, 2])
            t_list.append(np.squeeze(t_unique))
            assert len(t_unique) == 1

    if FLAGS.pde == 'pressurefree_stokes':
        assert len(fenics_functions) == 1
        key, subkey = jax.random.split(key)
        ground_truth = fenics_functions[0]
        points = pde.sample_points(subkey,
                                   3 * FLAGS.validation_points,
                                   jax_tools.tree_unstack(gt_params)[0])
        points = np.concatenate(points)
        taylor_fn = pde.SecondOrderTaylorLookup(ground_truth, points, d=3)
        fa_p = pde.get_p(taylor_fn)

    else:
        fa_p = None

    # --------------------- Run MAML --------------------

    time_last_log = time.time()
    for step in range(FLAGS.outer_steps):
        key, subkey = jax.random.split(key)
        with Timer() as t:
            optimizer, loss, loss_aux, grad_norm, bc_weights = train_step(subkey, optimizer, None)

        # increase NN domain every 100k steps or when we stop seeing val loss improvement
        if (FLAGS.pde == 'td_burgers') and (propagate_time) and (FLAGS.tmax_nn < FLAGS.tmax):
            FLAGS.tmax_nn += (FLAGS.tmax - FLAGS.tmin) / FLAGS.num_tsteps
            FLAGS.tmax_nn = np.clip(FLAGS.tmax_nn, a_max=FLAGS.tmax).astype(float)
            log(f"Passing new t max to NN: {FLAGS.tmax_nn}")
            logging.info(f"Passing new t max to NN: {FLAGS.tmax_nn}")
            last_prop_step = step
            propagate_time = False
            sys.stdout.flush()

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
            if FLAGS.annealing:
                bc_weights = {k: float(v) for k, v in bc_weights.items()}
                log(f"bc weigths for annealing (l2={FLAGS.annealing_l2}): ", bc_weights)
                if step > 0:
                    assert(bc_weights != None )

            if tflogger is not None:
                for k in loss_vals_and_grad_norms:
                    tflogger.log_scalar(
                        "grad_norm_{}".format(k),
                        float(loss_vals_and_grad_norms[k][1]),
                        step,
                    )
                _k1, _k2 = jax.random.split(
                    jax.random.split(subkey, FLAGS.bsize)[0], 2
                )
                _params = jax_tools.tree_unstack(gt_params)[0]
                _points = pde.sample_points(_k2, FLAGS.outer_points, _params)
                plt.figure()
                for _pointsi in _points:
                    plt.scatter(_pointsi[:, 0], _pointsi[:, 1], label=f'n_points={_pointsi.shape[0]}')
                    plt.legend()
                tflogger.log_plots("Points", [plt.gcf()], step)
                _all_points = np.concatenate(_points)
                _vals = optimizer.target(_all_points)
                _vals = _vals.reshape((_vals.shape[0], -1))
                _boundary_losses, _domain_losses = jax.vmap(
                    lambda x: pde.loss_fn(
                        optimizer.target,
                        (x.reshape(1, -1) for _ in range(len(_points))),
                        _params,
                    )
                )(_all_points)
                _all_losses = {**_boundary_losses, **_domain_losses}
                for _losskey in _all_losses:
                    # print(_losskey)
                    plt.figure()
                    _loss = _all_losses[_losskey]
                    # print(_loss.shape)
                    while len(_loss.shape) > 1:
                        _loss = _loss.mean(axis=1)
                    clrs = plt.scatter(
                        _all_points[:, 0],
                        _all_points[:, 1],
                        c=_loss[: len(_all_points)],
                    )
                    plt.colorbar(clrs)
                    tflogger.log_plots("{}".format(_losskey), [plt.gcf()], step)

                for dim in range(_vals.shape[1]):
                    plt.figure()
                    clrs = plt.scatter(
                        _all_points[:, 0], _all_points[:, 1], c=_vals[:, dim]
                    )
                    plt.colorbar(clrs)
                    tflogger.log_plots(
                        "Outputs dim {}".format(dim), [plt.gcf()], step
                    )
                _outputs_on_coords = optimizer.target(coords[0])

                for dim in range(_vals.shape[1]):
                    plt.figure()
                    clrs = plt.scatter(
                        coords[0][:, 0], coords[0][:, 1], c=_outputs_on_coords[:, dim]
                    )
                    plt.colorbar(clrs)
                    tflogger.log_plots(
                        "NN_on_coords dim {}".format(dim), [plt.gcf()], step
                    )

                    plt.figure()
                    clrs = plt.scatter(
                        coords[0][:, 0], coords[0][:, 1], c=fenics_vals[0][:, dim]
                    )
                    plt.colorbar(clrs)
                    tflogger.log_plots(
                        "Fenics_on_coords dim {}".format(dim), [plt.gcf()], step
                    )

                    to_plot = fenics_vals[0][:, dim] - _outputs_on_coords[:, dim]
                    plt.figure()
                    clrs = plt.scatter(
                        coords[0][:, 0], coords[0][:, 1], c=to_plot
                    )
                    plt.colorbar(clrs)
                    tflogger.log_plots(
                        "Residual_on_coords dim {}".format(dim), [plt.gcf()], step
                    )



        if step % FLAGS.log_every == 0:
            with Timer() as deploy_timer:
                mse, norms, rel_err, per_dim_rel_err, rel_err_std, t_rel_sq_err = trainer_util.vmap_validation_error(
                    optimizer.target, gt_params, coords, fenics_vals, make_coef_func
                )
                mse.block_until_ready()
            deployment_time = deploy_timer.interval / FLAGS.n_eval

            val_loss = validation_losses(optimizer.target)

            if FLAGS.pde == 'td_burgers' and len(early_stopping_tracker) == 3:
                improve_pct = (npo.mean(early_stopping_tracker) - val_loss) / npo.mean(early_stopping_tracker)
                _ = early_stopping_tracker.popleft()
                if (improve_pct < FLAGS.propagatetime_rel) and (step - last_prop_step) >= (FLAGS.propagatetime_max // 2):
                    propagate_time = True
                elif (step - last_prop_step) >= FLAGS.propagatetime_max:
                    propagate_time = True
            if FLAGS.pde == 'td_burgers':
                early_stopping_tracker.append(val_loss)

        # if step % FLAGS.log_every == 0:
            if step > 0:
                log("time {} steps: {}".format(FLAGS.log_every,
                                               time.time() - time_last_log))
            time_last_log = time.time()

            log(
                "step: {}, loss: {}, val_loss: {}, val_mse: {}, "
                "val_rel_err: {}, val_rel_err_std: {}, val_true_norms: {}, "
                "per_dim_rel_err: {}, deployment_time: {}, grad_norm: {}, per_time_step_error: {} ,time: {}".format(
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

                tflogger.log_scalar("NN_tmax", float(FLAGS.tmax_nn), step)

                if step % FLAGS.viz_every == 0:
                    # These take lots of filesize so only do them sometimes

                    for k, v in jax_tools.dict_flatten(optimizer.target.params):
                        tflogger.log_histogram("Param: " + k, v.flatten(), step)
                        if 'scale' in k:
                            print("Scale params: {}: {}".format(k, v))


            log("time for logging {}".format(time.time() - time_last_log))
            time_last_log = time.time()

        if FLAGS.viz_every > 0 and step % FLAGS.viz_every == 0:
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

            if tflogger is not None:
                tflogger.log_plots("Ground truth comparison", [plt.gcf()], step)

            if FLAGS.pde == 'td_burgers':
                tmp_filenames = trainer_util.plot_model_time_series(
                    optimizer.target,
                    pde,
                    fenics_functions,
                    gt_params,
                    get_final_model,
                    None,
                    0,
                )
                gif_out = os.path.join(path, "td_burger_step_{}.gif".format(step))
                pde.build_gif(tmp_filenames, outfile=gif_out)

    #if FLAGS.expt_name is not None:
    #    outfile.close()

    plt.figure()
    trainer_util.compare_plots_with_ground_truth(
        optimizer.target, pde, fenics_functions, gt_params, get_final_model, None, 0,
    )
    if FLAGS.expt_name is not None:
        plt.savefig(os.path.join(path, "viz_step_{}.png".format(step)), dpi=800)
        plt.savefig(os.path.join(path, "viz_final.png"), dpi=800)
    else:
        plt.show()


if __name__ == "__main__":
    app.run(main)
