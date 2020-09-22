from jax.config import config

import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from ..nets.field import NeuralPotential
from ..nets import maml

from functools import partial
import flax
from flax import nn
import fenics as fa

from .poisson_fenics import solve_fenics
from .poisson_common import *
from .poisson_common import loss_fn as base_loss_fn
from ..util import pcgrad
from ..util.timer import Timer

import matplotlib.pyplot as plt
import pdb
import sys
import os

import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--bsize", type=int, default=16, help="batch size (in tasks)")
parser.add_argument("--n_eval", type=int, default=16, help="num eval tasks")
parser.add_argument("--inner_lr", type=float, default=3e-5, help="inner learning rate")
parser.add_argument("--outer_lr", type=float, default=2e-4, help="outer learning rate")
parser.add_argument(
    "--outer_points",
    type=int,
    default=512,
    help="num query points on the boundary and in domain",
)
parser.add_argument(
    "--inner_points",
    type=int,
    default=512,
    help="num support points on the boundary and in domain",
)
parser.add_argument("--inner_steps", type=int, default=10, help="num inner steps")
parser.add_argument("--outer_steps", type=int, default=int(1e6), help="num outer steps")
parser.add_argument("--num_layers", type=int, default=5, help="num fcnn layers")
parser.add_argument("--layer_size", type=int, default=64, help="fcnn layer size")
parser.add_argument("--vary_source", type=int, default=1, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=1, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=1, help="1=true.")
parser.add_argument("--siren", type=int, default=1, help="1=true.")
parser.add_argument("--pcgrad", type=float, default=0.0, help="1=true.")
parser.add_argument("--bc_weight", type=float, default=1e1, help="weight on bc loss")
parser.add_argument("--out_dir", type=str, default="poisson_meta_results")
parser.add_argument("--expt_name", type=str, default=None)
parser.add_argument(
    "--viz_every", type=int, default=int(1e3), help="plot every N steps"
)


if __name__ == "__main__":
    args = parser.parse_args()


    if args.expt_name is not None:
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
        path = os.path.join(args.out_dir, args.expt_name)
        if os.path.exists(path):
            os.remove(path)
        outfile = open(path, "w")

        def log(*args, **kwargs):
            print(*args, **kwargs, flush=True)
            print(*args, **kwargs, file=outfile, flush=True)

    else:

        def log(*args, **kwargs):
            print(*args, **kwargs, flush=True)


    # --------------------- Defining the meta-training algorithm --------------------

    def loss_fn(
        potential_fn, points_on_boundary, points_in_domain, source_params, bc_params
    ):
        loss_on_boundary, loss_in_domain = base_loss_fn(
            points_on_boundary, points_in_domain, potential_fn, source_params, bc_params
        )

        return args.bc_weight * loss_on_boundary + loss_in_domain

    def make_task_loss_fns(key):
        # The input key is terminal
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        source_params, bc_params, geo_params = sample_params(k1, args)
        outer_in_domain = sample_points_in_domain(k2, args.outer_points, geo_params)
        outer_on_boundary = sample_points_on_boundary(
            k3, args.outer_points, geo_params
        )
        inner_in_domain = sample_points_in_domain(k4, args.inner_points, geo_params)
        inner_on_boundary = sample_points_on_boundary(k5, args.inner_points, geo_params)

        inner_loss = lambda key, potential_fn: loss_fn(
            potential_fn, inner_on_boundary, inner_in_domain, source_params, bc_params
        )
        outer_loss = lambda key, potential_fn: loss_fn(
            potential_fn, outer_on_boundary, outer_in_domain, source_params, bc_params
        )
        return inner_loss, outer_loss

    make_inner_opt = flax.optim.Momentum(learning_rate=args.inner_lr, beta=0.).create

    maml_def = maml.MamlDef(
        make_inner_opt=make_inner_opt,
        make_task_loss_fns=make_task_loss_fns,
        inner_steps=args.inner_steps,
        n_batch_tasks=args.bsize,
    )

    Potential = NeuralPotential.partial(
        sizes=[args.layer_size for _ in range(args.num_layers)],
        dense_args=(),
        nonlinearity=np.sin if args.siren else nn.swish,
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    _, init_params = Potential.init_by_shape(subkey, [((1, 2), DTYPE)])
    optimizer = flax.optim.Adam(learning_rate=args.outer_lr).create(
        flax.nn.Model(Potential, init_params))


    # --------------------- Defining the evaluation functions --------------------

    def get_ground_truth_points(source_params_list, bc_params_list, geo_params_list):
        fenics_functions = []
        potentials = []
        coords = []
        for sp, bp, gp in zip(source_params_list, bc_params_list, geo_params_list):
            c1, c2 = gp
            ground_truth = solve_fenics(sp, bp, gp)
            fenics_functions.append(ground_truth)
            potentials.append(np.array(ground_truth.vector()[:]))
            coords.append(
                np.array(ground_truth.function_space().tabulate_dof_coordinates())
            )
        return fenics_functions, potentials, coords

    @jax.jit
    def make_potential_func(
        key, model, source_params, bc_params, geo_params, coords
    ):
        # Input key is terminal
        k1, k2, k3 = jax.random.split(key, 3)
        inner_in_domain = sample_points_in_domain(k1, args.inner_points, geo_params)
        inner_on_boundary = sample_points_on_boundary(k2, args.inner_points, geo_params)
        inner_loss_fn = lambda key, potential_fn: loss_fn(
            potential_fn, inner_on_boundary, inner_in_domain, source_params, bc_params
        )
        final_model, _ = maml.single_task_rollout(maml_def, k3, model, inner_loss_fn)
        return np.squeeze(final_model(coords))

    def compare_plots_with_ground_truth(
        model, fenics_functions, sources, bc_params, geo_params
    ):
        keys = jax.random.split(jax.random.PRNGKey(0), len(fenics_functions))
        M = len(fenics_functions)
        for j in range(int(np.ceil(M / 10))):
            ffs = fenics_functions[j * 10 : (j + 1) * 10]
            N = len(ffs)
            assert N % 2 == 0
            for i in range(N):
                ground_truth = fenics_functions[i]
                # top two rows are optimizer.target
                potentials = make_potential_func(
                    keys[i],
                    model,
                    sources[i],
                    bc_params[i],
                    geo_params[i],
                    ground_truth.function_space().tabulate_dof_coordinates(),
                )

                u_approx = fa.Function(ground_truth.function_space())
                potentials.reshape(np.array(u_approx.vector()[:]).shape)
                u_approx.vector().set_local(potentials)

                plt.subplot(2, N, 1 + i)

                fa.plot(u_approx, title="Approx")
                # bottom two rots are ground truth
                plt.subplot(2, N, 1 + i + N)
                fa.plot(ground_truth, title="Truth")


    @partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
    def vmap_validation_error(
        model,
        ground_truth_source,
        ground_truth_bc,
        ground_truth_geo,
        trunc_coords,
        trunc_true_potentials,
    ):
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, args.n_eval)
        potentials = vmap(make_potential_func, (0, None, 0, 0, 0, 0))(
            keys,
            model,
            ground_truth_source,
            ground_truth_bc,
            ground_truth_geo,
            trunc_coords,
        )

        return np.mean((potentials - trunc_true_potentials) ** 2)


    @jax.jit
    def validation_losses(model):
        _, losses, meta_losses = maml.multi_task_grad_and_losses(
            maml_def, jax.random.PRNGKey(0), model,
        )
        return losses, meta_losses


    grid = npo.zeros([101, 101, 2])
    for i in range(101):
        for j in range(101):
            grid[i, j] += [i * 1.0 / 100, j * 1.0 / 100]
    grid = np.array(grid, dtype=DTYPE).reshape(-1, 2) * 2 - np.array(
        [[1.0, 1.0]], dtype=DTYPE
    )

    assert args.n_eval % 2 == 0

    key, gt_key = jax.random.split(key, 2)

    gt_keys = jax.random.split(gt_key, args.n_eval)
    ground_truth_source, ground_truth_bc, ground_truth_geo = vmap(
        sample_params, (0, None)
    )(gt_keys, args)

    fenics_functions, fenics_potentials, coords = get_ground_truth_points(
        ground_truth_source, ground_truth_bc, ground_truth_geo
    )

    trunc_true_potentials = np.stack(
        [p[: min([len(p) for p in fenics_potentials])] for p in fenics_potentials]
    )

    trunc_coords = np.stack([c[: min([len(c) for c in coords])] for c in coords])



    # --------------------- Run MAML --------------------

    for step in range(args.outer_steps):
        key, subkey = jax.random.split(key, 2)

        with Timer() as t:
            meta_grad, losses, meta_losses = maml.multi_task_grad_and_losses(
                maml_def, subkey, optimizer.target,
            )
            meta_grad_norm = np.sqrt(jax.tree_util.tree_reduce(lambda x, y: x+y, 
                jax.tree_util.tree_map(lambda x: np.sum(x**2), meta_grad)))
            if np.isfinite(meta_grad_norm):
                if meta_grad_norm > 1.:
                    log("clipping gradients with norm {}".format(
                        meta_grad_norm))
                    meta_grad = jax.tree_util.tree_map(
                        lambda x: x/meta_grad_norm, meta_grad)  
                optimizer = optimizer.apply_gradient(meta_grad)

        val_error = vmap_validation_error(
            optimizer.target,
            ground_truth_source,
            ground_truth_bc,
            ground_truth_geo,
            trunc_coords,
            trunc_true_potentials,
        )

        val_losses, val_meta_losses = validation_losses(optimizer.target)

        log(
            "step: {}, meta_loss: {}, val_meta_loss: {}, val_err: {}, "
            "meta_grad_norm: {}, time: {}".format(
                step, np.mean(meta_losses), np.mean(val_meta_losses),
                val_error, meta_grad_norm, t.interval
            )
        )
        log("meta_loss_max: {}, meta_loss_min: {}, meta_loss_std: {}".format(
            np.max(meta_losses), np.min(meta_losses), np.std(meta_losses)))
        log(
            "per_step_losses: {}\nper_step_val_losses:{}\n".format(
                np.mean(losses, axis=0), np.mean(val_losses, axis=0),
            )
        )

        if args.viz_every > 0 and step % args.viz_every == 0:
            plt.figure()
            compare_plots_with_ground_truth(
                optimizer.target,
                fenics_functions,
                ground_truth_source,
                ground_truth_bc,
                ground_truth_geo,
            )
            if args.expt_name is not None:
                plt.savefig(
                    os.path.join(
                        args.out_dir, args.expt_name + "_viz_step_{}.png".format(step)
                    )
                )
            else:
                plt.show()

    if args.expt_name is not None:
        outfile.close()

    plt.figure()
    compare_plots_with_ground_truth(
        optimizer.target,
        fenics_functions,
        ground_truth_source,
        ground_truth_bc,
        ground_truth_geo,
    )
    if args.expt_name is not None:
        plt.savefig(os.path.join(args.out_dir, args.expt_name + "_viz_final.png"))
    else:
        plt.show()
