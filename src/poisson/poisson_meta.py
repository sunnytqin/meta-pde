import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from ..nets.field import NeuralPotential
from ..nets.gradient_conditioned import GradientConditionedField


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

from jax.config import config
config.enable_omnistaging()

parser = argparse.ArgumentParser()
parser.add_argument("--bsize", type=int, default=8, help="batch size (in tasks)")
parser.add_argument("--n_eval", type=int, default=8, help="num eval tasks")
parser.add_argument("--inner_lr", type=float, default=1e-4, help="inner learning rate")
parser.add_argument("--outer_lr", type=float, default=3e-4, help="outer learning rate")
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
parser.add_argument("--inner_steps", type=int, default=2, help="num inner steps")
parser.add_argument("--outer_steps", type=int, default=int(1e6), help="num outer steps")
parser.add_argument("--num_layers", type=int, default=5, help="num fcnn layers")
parser.add_argument("--layer_size", type=int, default=64, help="fcnn layer size")
parser.add_argument("--vary_source", type=int, default=1, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=1, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=1, help="1=true.")
parser.add_argument("--siren", type=int, default=0, help="1=true.")
parser.add_argument("--pcgrad", type=float, default=0.0, help="1=true.")
parser.add_argument("--bc_weight", type=float, default=1e1, help="weight on bc loss")
parser.add_argument("--out_dir", type=str, default="poisson_meta_results")
parser.add_argument("--expt_name", type=str, default=None)
parser.add_argument("--viz_every", type=int, default=int(1e3),
                    help="plot every N steps")


if __name__ == '__main__':
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


    def loss_fn(potential_fn, points_on_boundary, points_in_domain,
                source_params, bc_params):
        loss_on_boundary, loss_in_domain = base_loss_fn(
            points_on_boundary, points_in_domain, potential_fn, source_params, bc_params)

        return args.bc_weight * loss_on_boundary + loss_in_domain


    def get_meta_loss(
        points_in_domain,
        points_on_boundary,
        inner_in_domain,
        inner_on_boundary,
        source_params,
        bc_params,
    ):
        potential_fn = lambda model: partial(
            model,
            inner_loss_kwargs={
                "points_in_domain": inner_in_domain,
                "points_on_boundary": inner_on_boundary,
                "source_params": source_params,
                "bc_params": bc_params,
            },
        )

        def meta_loss(model):
            return loss_fn(
                points_on_boundary=points_on_boundary,
                points_in_domain=points_in_domain,
                source_params=source_params,
                bc_params=bc_params,
                potential_fn=potential_fn(model),
            )

        return meta_loss


    def get_single_example_loss(args, key):
        # The input key is terminal
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        source_params, bc_params, geo_params = sample_params(k1, args)
        points_in_domain = sample_points_in_domain(k2, args.outer_points, geo_params)
        points_on_boundary = sample_points_on_boundary(k3, args.outer_points, geo_params)
        inner_in_domain = sample_points_in_domain(k4, args.inner_points, geo_params)
        inner_on_boundary = sample_points_on_boundary(k5, args.inner_points, geo_params)

        return get_meta_loss(
            points_in_domain=points_in_domain,
            points_on_boundary=points_on_boundary,
            inner_in_domain=inner_in_domain,
            inner_on_boundary=inner_on_boundary,
            source_params=source_params,
            bc_params=bc_params,
        )


    def get_batch_loss(args, keys):
        def loss(args, model, key):
            return get_single_example_loss(args, key)(model)

        def batch_loss(model):
            return jax.vmap(partial(loss, args, model))(keys)

        return batch_loss


    # @partial(jax.jit, static_argnums=(0,))
    def batch_train_step(args, optimizer, key):
        # The input key is terminal
        keys = jax.random.split(key, args.bsize)
        batch_loss_fn = get_batch_loss(args, keys)
        loss, gradient = jax.value_and_grad(lambda m: batch_loss_fn(m).mean())(
            optimizer.target
        )
        optimizer = optimizer.apply_gradient(gradient)
        return optimizer, loss


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


    def make_potential_func(key, optimizer,
                            source_params, bc_params, geo_params, coords, args):
        # Input key is terminal
        k1, k2 = jax.random.split(key)
        inner_in_domain = sample_points_in_domain(k1, args.inner_points, geo_params)
        inner_on_boundary = sample_points_on_boundary(k2, args.inner_points, geo_params)
        potential_fn = partial(
            optimizer.target,
            inner_loss_kwargs={
                "points_in_domain": inner_in_domain,
                "points_on_boundary": inner_on_boundary,
                "source_params": source_params,
                "bc_params": bc_params,
            },
        )
        return np.squeeze(potential_fn(coords))


    def compare_plots_with_ground_truth(
        optimizer, fenics_functions, sources, bc_params, geo_params, args
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
                    optimizer,
                    sources[i],
                    bc_params[i],
                    geo_params[i],
                    ground_truth.function_space().tabulate_dof_coordinates(),
                    args,
                )

                u_approx = fa.Function(ground_truth.function_space())
                potentials.reshape(np.array(u_approx.vector()[:]).shape)
                u_approx.vector().set_local(potentials)

                plt.subplot(4, N // 2, 1 + i)

                fa.plot(u_approx, title="Approx")
                # bottom two rots are ground truth
                plt.subplot(4, N // 2, 1 + N + i)
                fa.plot(ground_truth, title="Truth")


    def vmap_validation_error(
        optimizer,
        ground_truth_source,
        ground_truth_bc,
        ground_truth_geo,
        trunc_coords,
        trunc_true_potentials,
        args,
    ):
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, args.n_eval)
        potentials = vmap(make_potential_func, (0, None, 0, 0, 0, 0, None))(
            keys,
            optimizer,
            ground_truth_source,
            ground_truth_bc,
            ground_truth_geo,
            trunc_coords,
            args,
        )

        return np.mean((potentials - trunc_true_potentials) ** 2)


    GCField = GradientConditionedField.partial(
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        train_inner_lrs=True,
        inner_loss=loss_fn,
        # n_fourier=5,
        base_args={
            "sizes": [args.layer_size for _ in range(args.num_layers)],
            "input_dim": 2,
            "output_dim": 1,
            "nonlinearity": np.sin if args.siren else nn.swish,
            "kernel_init": flax.nn.initializers.variance_scaling(
                2.0, "fan_in", "truncated_normal"
            ),
        },  # These base_args are the arguments for the initialization of GradientConditionedFieldParameters
    )


    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    _, init_params = GCField.partial(
        inner_loss_kwargs={}, inner_loss=lambda x: 0.0
    ).init_by_shape(jax.random.PRNGKey(0), [((1, 2), DTYPE)])

    model = flax.nn.Model(
        GCField, init_params
    )

    optimizer = flax.optim.Adam(learning_rate=args.outer_lr).create(model)


    grid = npo.zeros([101, 101, 2])
    for i in range(101):
        for j in range(101):
            grid[i, j] += [i * 1.0 / 100, j * 1.0 / 100]
    grid = np.array(grid, dtype=DTYPE).reshape(-1, 2) * 2 - np.array([[1.0, 1.0]],
                                                                     dtype=DTYPE)

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


    for step in range(args.outer_steps):
        key, subkey = jax.random.split(key, 2)
        with Timer() as t:
            optimizer, loss = batch_train_step(args, optimizer, subkey)
            loss = float(loss)

        val_error = vmap_validation_error(
            optimizer,
            ground_truth_source,
            ground_truth_bc,
            ground_truth_geo,
            trunc_coords,
            trunc_true_potentials,
            args,
        )

        log("step: {}, loss: {}, val: {}, time: {}".format(step, loss,
                                                           val_error, t.interval))

        if args.viz_every > 0 and step % args.viz_every == 0:
            plt.figure()
            compare_plots_with_ground_truth(
                optimizer, fenics_functions,
                ground_truth_source, ground_truth_bc, ground_truth_geo, args
            )
            if args.expt_name is not None:
                plt.savefig(os.path.join(args.out_dir, args.expt_name +
                                         "_viz_step_{}.png".format(step)))
            else:
                plt.show()


    if args.expt_name is not None:
        outfile.close()

    plt.figure()
    compare_plots_with_ground_truth(
        optimizer, fenics_functions,
        ground_truth_source, ground_truth_bc, ground_truth_geo, args
    )
    if args.expt_name is not None:
        plt.savefig(os.path.join(args.out_dir, args.expt_name +
                                 "_viz_final.png"))
    else:
        plt.show()
