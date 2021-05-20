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
from collections import defaultdict
import pdb
import sys
import os
from copy import deepcopy
from collections import namedtuple

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
        loss = np.sum(
            np.array([bl for bl in boundary_losses.values()])
        ) + np.sum(
            np.array([dl for dl in domain_losses.values()]))

        # return the total loss, and as aux a dict of individual losses
        return loss, {**boundary_losses, **domain_losses}

    def task_loss_fn(key, model):
        # The input key is terminal
        k1, k2 = jax.random.split(key, 2)
        params = pde.sample_params(k1)
        points = pde.sample_points(k2, FLAGS.outer_points, params)
        return loss_fn(model, points, params)

    @jax.jit
    def batch_loss_fn(key, model, bc_weights_prev):
        vmap_task_loss_fn = jax.vmap(task_loss_fn, (0, None))

        keys = jax.random.split(key, FLAGS.bsize)
        _, loss_dict = vmap_task_loss_fn(keys, model)

        loss_aux = {}  # store original loss by loss type
        loss_grads = {}  # store original loss gradient by loss type

        # get original gradients
        for k in loss_dict:
            loss_aux[k] = np.sum(loss_dict[k])
            # do nothing if not annealing and not pcgrad-ing
            if ((not FLAGS.annealing and not FLAGS.pcgrad) \
                    or bc_weights_prev is None):
                continue
            single_loss_fn = lambda model: np.sum(vmap_task_loss_fn(keys, model)[1][k])
            _, loss_grad = jax.value_and_grad(single_loss_fn)(model)
            loss_grads[k] = loss_grad

        # perform PC grad
        if FLAGS.pcgrad and bc_weights_prev is not None:
            # place holder for new total loss gradient
            total_grad_new = [np.zeros_like(x)
                              for x in jax.tree_flatten(loss_grad)[0]]
            _, loss_grads = perform_pcgrad(loss_grads, total_grad_new)

        # perform weight annealing
        bc_weights = defaultdict(lambda: 1.0)
        if FLAGS.annealing and bc_weights_prev is not None:
            bc_weights = perform_annealing(loss_grads, bc_weights_prev)

        # recompute loss
        loss = np.sum(np.array([bc_weights[k] * v for k, v in loss_aux.items()]))

        return loss, (loss_aux, bc_weights)

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
    def perform_pcgrad(loss_grads, total_grad_new):
        # define projection function
        project = partial(pcgrad.project_grads, FLAGS.pcgrad_norm)

        # for new individual loss gradient
        loss_grads_new = {}

        # perform PC grad for each loss type
        for k in loss_grads:
            grad = loss_grads[k]
            other_grads = {k1: v1 for k1, v1 in loss_grads.items() if k1 != k}
            loss_grad_new = jax.tree_multimap(
                project, grad, *other_grads.values()
            )
            # update the new projected individual loss function
            loss_grads_new[k] = loss_grad_new
            loss_grad_new_flat, treedef = jax.tree_flatten(loss_grad_new)
            # update gradient for the entire loss function
            total_grad_new = [x + y for x, y in zip(loss_grad_new_flat, total_grad_new)]


        total_grad_new = jax.tree_unflatten(
            treedef, total_grad_new
        )

        return total_grad_new, loss_grads_new

    @jax.jit
    def perform_annealing(loss_grads, bc_weights_prev):
        for k, loss_grad in loss_grads.items():
            if k == FLAGS.domain_loss:
                domain_loss_grad_max = np.max(
                    np.array(jax.tree_flatten(
                        jax.tree_util.tree_map(lambda x: np.sum(np.abs(x)), loss_grad))[0]
                             )
                )
            else:
                loss_grad_mean = np.mean(
                    np.array(jax.tree_flatten(
                        jax.tree_util.tree_map(lambda x: np.sum(np.abs(x)), loss_grad))[0]
                             )
                )
                bc_weights[k] = loss_grad_mean
        for k in loss_grads:
            if k == FLAGS.domain_loss:
                continue
            bc_weights[k] = (1 - FLAGS.annealing_alpha) * bc_weights_prev[k] + \
                            FLAGS.annealing_alpha * (domain_loss_grad_max / bc_weights[k])
        return bc_weights

    Field = pde.BaseField.partial(
        sizes=[FLAGS.layer_size for _ in range(FLAGS.num_layers)],
        dense_args=(),
        nonlinearity=np.sin if FLAGS.siren else nn.swish,
        omega=FLAGS.siren_omega,
        omega0=FLAGS.siren_omega0,
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    _, init_params = Field.init_by_shape(subkey, [((1, 2), np.float32)])

    optimizer = trainer_util.get_optimizer(Field, init_params)

    bc_weights = defaultdict(lambda: 1.0)

    # --------------------- Defining the evaluation functions --------------------


    def get_final_model(unused_key, model,
                         _unused_params=None,
                         _unused_num_steps=None,
                         _unused_meta_alg_def=None):
        # Input key is terminal
        return model

    @jax.jit
    def make_coef_func(key, model, params, coords):
        # Input key is terminal
        final_model = get_final_model(key, model, params)

        return np.squeeze(final_model(coords))



    @jax.jit
    def train_step(key, optimizer, bc_weights_prev):

        if FLAGS.optimizer == "adahessian":
            k1, k2 = jax.random.split(key)
            loss, (loss_aux, bc_weights) = batch_loss_fn(k1, optimizer.target, bc_weights_prev)
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

    gt_keys = jax.random.split(gt_key, FLAGS.n_eval)
    gt_params = vmap(pde.sample_params)(gt_keys)
    print("gt_params: {}".format(gt_params))

    fenics_functions, fenics_vals, coords = trainer_util.get_ground_truth_points(
        pde, jax_tools.tree_unstack(gt_params), gt_points_key
    )

    # --------------------- Run MAML --------------------

    time_last_log = time.time()

    for step in range(FLAGS.outer_steps):
        key, subkey = jax.random.split(key)
        with Timer() as t:
            optimizer, loss, loss_aux, grad_norm, bc_weights = train_step(subkey, optimizer, bc_weights)
        # ---- This big section is logging a bunch of debug stats
        # loss grad norms; plotting the sampled points; plotting the vals at those
        # points; plotting the losses at those points.

        # Todo (alex) -- see if we can clean it up, and maybe also do it in maml etc
        if (
            FLAGS.measure_grad_norm_every > 0
            and step % FLAGS.measure_grad_norm_every == 0
        ):
            loss_vals_and_grad_norms = get_grad_norms(subkey, optimizer.target)
            loss_vals_and_grad_norms = {k: (float(v[0]), float(v[1]))
                                        for k, v in loss_vals_and_grad_norms.items()}
            log("loss vals and grad norms: ", loss_vals_and_grad_norms)
            if FLAGS.annealing:
                bc_weights = {k: float(v)for k, v in bc_weights.items()}
                log("bc weigths for annealing: ", bc_weights)
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
                _params = pde.sample_params(_k1)
                _points = pde.sample_points(_k2, FLAGS.outer_points, _params)
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

        if step % FLAGS.val_every == 0:
            with Timer() as deploy_timer:
                mse, norms, rel_err, per_dim_rel_err, rel_err_std = trainer_util.vmap_validation_error(
                    optimizer.target, gt_params, coords, fenics_vals, make_coef_func
                )
                mse.block_until_ready()
            deployment_time = deploy_timer.interval / FLAGS.n_eval

            val_loss = validation_losses(optimizer.target)

        if step % FLAGS.log_every == 0:
            if step > 0:
                log("time {} steps: {}".format(FLAGS.log_every,
                                               time.time() - time_last_log))
            time_last_log = time.time()

            log(
                "step: {}, loss: {}, val_loss: {}, val_mse: {}, "
                "val_rel_err: {}, val_rel_err_std: {}, val_true_norms: {}, "
                "per_dim_rel_err: {}, deployment_time: {}, grad_norm: {}, time: {}".format(
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

    #if FLAGS.expt_name is not None:
    #    outfile.close()

    plt.figure()
    trainer_util.compare_plots_with_ground_truth(
        optimizer.target, pde, fenics_functions, gt_params, get_final_model, None, 0,
    )
    if FLAGS.expt_name is not None:
        plt.savefig(os.path.join(path, "viz_final.png"), dpi=800)
    else:
        plt.show()


if __name__ == "__main__":
    app.run(main)
