import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap
from flax import serialization


from ..nets.field import NeuralField
from ..nets import maml

from functools import partial
import flax
from flax import nn
from ..util.timer import Timer
from ..util import pcgrad

from .metamaterial_fenics import solve_fenics, make_fenics
from .metamaterial_common import *
import fenics as fa

import matplotlib.pyplot as plt
import pdb

import os
import sys

import pickle

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--bsize", type=int, default=2, help="batch size (in tasks)")
parser.add_argument("--n_eval", type=int, default=2, help="num eval tasks")
parser.add_argument("--max_plot", type=int, default=6, help="num eval tasks to plot")
parser.add_argument(
    "--lr_inner_lr", type=float, default=1e-2, help="lr for inner learning rate"
)
parser.add_argument("--inner_lr", type=float, default=1e-3, help="inner learning rate")
parser.add_argument("--outer_lr", type=float, default=3e-5, help="outer learning rate")
parser.add_argument(
    "--outer_points",
    type=int,
    default=32,
    help="num query points on the boundary and in domain",
)
parser.add_argument(
    "--inner_points",
    type=int,
    default=32,
    help="num support points on the boundary and in domain",
)
parser.add_argument("--gridsize", type=int, default=128, help="gridsize for sampling")
parser.add_argument("--inner_steps", type=int, default=2, help="num inner steps")
parser.add_argument("--outer_steps", type=int, default=int(1e4), help="num outer steps")
parser.add_argument("--num_layers", type=int, default=5, help="num fcnn layers")
parser.add_argument("--n_fourier", type=int, default=None, help="num fourier features")
parser.add_argument("--layer_size", type=int, default=64, help="fcnn layer size")
parser.add_argument("--siren", type=int, default=0, help="1 for true")
parser.add_argument("--pcgrad", type=float, default=0.0, help="1=true.")
parser.add_argument("--vary_source", type=int, default=0, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=1, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=1, help="1=true.")
parser.add_argument(
    "--bc_scale", type=float, default=0.15, help="scale of bc displacement"
)
parser.add_argument(
    "--bc_weight", type=float, default=100.0, help="weight on outer boundary loss",
)
parser.add_argument(
    "--domain_weight", type=float, default=1e-5, help="weight on domain loss",
)
parser.add_argument(
    "--interior_weight",
    type=float,
    default=1e-4,
    help="weight on interior boundary loss",
)
parser.add_argument(
    "--n_cells", help="number cells on one side of ref volume", type=int, default=1
)
parser.add_argument("--out_dir", type=str, default="mm_meta_results")
parser.add_argument("--expt_name", type=str, default=None)
parser.add_argument("--load_ckpt_file", type=str, default=None)
parser.add_argument("--viz_every", type=int, default=10000, help="plot every N steps")


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

    log(args)

    def loss_fn(
        field_fn,
        points_on_boundary,
        points_in_domain,
        points_on_interior,
        source_params,
        bc_params,
        geo_params,
    ):

        boundary_loss = boundary_loss_fn(points_on_boundary, field_fn, bc_params)
        domain_loss = domain_loss_fn(points_in_domain, field_fn, source_params)
        interior_loss = interior_bc_loss_fn(
            points_on_interior, field_fn, geo_params, source_params
        )

        return (
            args.bc_weight * boundary_loss
            + args.domain_weight * domain_loss
            + args.interior_weight * interior_loss
        )

    def make_task_loss_fns(key):
        # The input key is terminal
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

        source_params, bc_params, geo_params = sample_params(k1, args)
        outer_in_domain = sample_points_in_domain(
            k2, args.outer_points, args.gridsize, geo_params
        )
        outer_on_boundary = sample_points_on_boundary(k3, args.outer_points)
        outer_on_interior = sample_points_on_interior_boundary(
            k4, args.outer_points, geo_params
        )
        inner_in_domain = sample_points_in_domain(
            k5, args.inner_points, args.gridsize, geo_params
        )
        inner_on_boundary = sample_points_on_boundary(k6, args.inner_points)
        inner_on_interior = sample_points_on_interior_boundary(
            k7, args.inner_points, geo_params
        )
        inner_loss = lambda key, field_fn: loss_fn(
            field_fn,
            inner_on_boundary,
            inner_in_domain,
            inner_on_interior,
            source_params,
            bc_params,
            geo_params,
        )
        outer_loss = lambda key, field_fn: loss_fn(
            field_fn,
            outer_on_boundary,
            outer_in_domain,
            outer_on_interior,
            source_params,
            bc_params,
            geo_params,
        )
        return inner_loss, outer_loss

    make_inner_opt = flax.optim.Momentum(learning_rate=args.inner_lr, beta=0.0).create

    maml_def = maml.MamlDef(
        make_inner_opt=make_inner_opt,
        make_task_loss_fns=make_task_loss_fns,
        inner_steps=args.inner_steps,
        n_batch_tasks=args.bsize,
    )

    inner_lrs = np.ones(maml_def.inner_steps)

    Field = NeuralField.partial(
        sizes=[args.layer_size for _ in range(args.num_layers)],
        dense_args=(),
        nonlinearity=np.sin if args.siren else nn.swish,
        n_fourier=args.n_fourier,
    )

    def get_ground_truth_points(source_params_list, bc_params_list, geo_params_list):
        fenics_functions = []
        true_fields = []
        coords = []
        for sp, bp, gp in zip(source_params_list, bc_params_list, geo_params_list):
            c1, c2 = gp
            ground_truth = solve_fenics(sp, bp, gp)
            fenics_functions.append(ground_truth)
            true_fields.append(np.array(ground_truth.vector()[:]).astype(DTYPE))
            coords.append(
                np.array(ground_truth.function_space().tabulate_dof_coordinates())[
                    ::2
                ].astype(DTYPE)
            )
        return fenics_functions, true_fields, coords

    @jax.jit
    def make_field_func(
        key, model_and_lrs, source_params, bc_params, geo_params, coords
    ):
        model, inner_lrs = model_and_lrs
        # Input key is terminal
        k1, k2, k3, k4 = jax.random.split(key, 4)
        inner_in_domain = sample_points_in_domain(
            k1, args.inner_points, args.gridsize, geo_params
        )
        inner_on_boundary = sample_points_on_boundary(k2, args.inner_points)
        inner_on_interior = sample_points_on_interior_boundary(
            k3, args.inner_points, geo_params
        )

        inner_loss_fn = lambda key, field_fn: loss_fn(
            field_fn,
            inner_on_boundary,
            inner_in_domain,
            inner_on_interior,
            source_params,
            bc_params,
            geo_params,
        )
        final_model, _ = maml.single_task_rollout(
            maml_def, k4, model, inner_loss_fn, inner_lrs
        )

        field_vals = np.squeeze(final_model(coords))
        # We have something of shape [n_dofs, 2] = Aij, i=1..n_dovs, j=1,2
        # We want a corresponding vector of scalars, shape [2*n_dofs]
        # Fenics represents this vector as
        #  [A00, A01, A10, A11, A20, A21 ...],
        # which is what we get by calling np.reshape(A, -1)
        field_vals = field_vals.reshape(-1)
        return field_vals

    def compare_plots_with_ground_truth(
        model_and_lrs, fenics_functions, sources, bc_params, geo_params
    ):
        keys = jax.random.split(jax.random.PRNGKey(0), len(fenics_functions))
        N = len(fenics_functions)
        if N > args.max_plot:
            N = args.max_plot
        assert N % 2 == 0
        for i in range(N):
            ground_truth = fenics_functions[i]
            # top two rows are optimizer.target
            field_vals = make_field_func(
                keys[i],
                model_and_lrs,
                sources[i],
                bc_params[i],
                geo_params[i],
                ground_truth.function_space().tabulate_dof_coordinates()[::2],
            ).astype(np.float32)

            u_approx = fa.Function(ground_truth.function_space())

            assert len(field_vals) == len(np.array(u_approx.vector()[:]))
            assert len(np.array(u_approx.vector()[:]).shape) == 1

            u_approx.vector().set_local(field_vals)

            u_diff = fa.Function(ground_truth.function_space())
            u_diff.vector().set_local(field_vals - np.array(ground_truth.vector()[:]))

            plt.subplot(3, N, 1 + i)

            fa.plot(u_approx, title="Approx", mode="displacement")
            # bottom two rots are ground truth
            plt.subplot(3, N, 1 + N + i)
            fa.plot(ground_truth, title="Truth", mode="displacement")

            plt.subplot(3, N, 1 + 2 * N + i)
            fa.plot(u_diff, title="Difference", mode="displacement")

    @jax.jit
    def vmap_validation_error(
        model_and_lrs,
        ground_truth_source,
        ground_truth_bc,
        ground_truth_geo,
        trunc_coords,
        trunc_true_fields,
    ):
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, args.n_eval)
        fields = vmap(make_field_func, (0, None, 0, 0, 0, 0))(
            keys,
            model_and_lrs,
            ground_truth_source,
            ground_truth_bc,
            ground_truth_geo,
            trunc_coords,
        )

        return np.sqrt(np.mean((fields - trunc_true_fields) ** 2))

    def save_opt(optimizer):
        state_bytes = serialization.to_bytes(optimizer)
        with open(os.path.join(args.out_dir, args.expt_name + "most_recent_state"),
                  'wb') as bytesfile:
            pickle.dump(state_bytes, bytesfile)

    def load_opt(optimizer):
        with open(args.load_ckpt_file, 'r') as bytesfile:
            state_bytes = pickle.load(bytesfile)
            optimizer = serialization.from_bytes(optimizer, state_bytes)
        return optimizer

    @jax.jit
    def validation_losses(model_and_lrs):
        _, losses, meta_losses = maml.multi_task_grad_and_losses(
            maml_def, jax.random.PRNGKey(0), model_and_lrs[0], model_and_lrs[1],
        )
        return losses, meta_losses

    assert args.n_eval % 2 == 0

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    _, init_params = Field.init_by_shape(subkey, [((1, 2), DTYPE)])

    optimizer = flax.optim.Adam(learning_rate=args.outer_lr).create(
        flax.nn.Model(Field, init_params)
    )

    if args.load_ckpt_file is not None:
        optimizer = load_opt(optimizer)

    key, gt_key = jax.random.split(key, 2)

    gt_keys = jax.random.split(gt_key, args.n_eval)

    ground_truth_source, ground_truth_bc, ground_truth_geo = vmap(
        sample_params, (0, None)
    )(gt_keys, args)

    fenics_functions, fenics_fields, coords = get_ground_truth_points(
        ground_truth_source, ground_truth_bc, ground_truth_geo
    )

    trunc_true_fields = np.stack(
        [p[: min([len(p) for p in fenics_fields])] for p in fenics_fields]
    )

    trunc_coords = np.stack([c[: min([len(c) for c in coords])] for c in coords])

    for step in range(args.outer_steps):
        key, subkey = jax.random.split(key, 2)

        with Timer() as t:
            meta_grad, losses, meta_losses = maml.multi_task_grad_and_losses(
                maml_def, subkey, optimizer.target, inner_lrs
            )
            meta_grad_norm = np.sqrt(
                jax.tree_util.tree_reduce(
                    lambda x, y: x + y,
                    jax.tree_util.tree_map(lambda x: np.sum(x ** 2), meta_grad),
                )
            )
            if np.isfinite(meta_grad_norm):
                if meta_grad_norm > 10.0:
                    log("clipping gradients with norm {}".format(meta_grad_norm))
                    meta_grad = jax.tree_util.tree_map(
                        lambda x: 10.0 * x / meta_grad_norm, meta_grad
                    )
                optimizer = optimizer.apply_gradient(meta_grad[0])
                inner_lrs = inner_lrs - args.lr_inner_lr * meta_grad[1]

        val_error = vmap_validation_error(
            (optimizer.target, inner_lrs),
            ground_truth_source,
            ground_truth_bc,
            ground_truth_geo,
            trunc_coords,
            trunc_true_fields,
        )

        val_losses, val_meta_losses = validation_losses((optimizer.target, inner_lrs))

        log(
            "step: {}, meta_loss: {}, val_meta_loss: {}, val_err: {}, meta_grad_norm: {}, time: {}".format(
                step,
                np.mean(meta_losses),
                np.mean(val_meta_losses),
                val_error,
                meta_grad_norm,
                t.interval,
            )
        )
        log(
            "per_step_losses: {}\nper_step_val_losses:{}\n".format(
                np.mean(losses, axis=0), np.mean(val_losses, axis=0),
            )
        )

        if args.viz_every > 0 and step % args.viz_every == 0:
            save_opt(optimizer)
            plt.figure()
            compare_plots_with_ground_truth(
                (optimizer.target, inner_lrs),
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
                plt.savefig(
                    os.path.join(
                        args.out_dir, args.expt_name + "_viz_step_{}.png".format(step)
                    )
                )
            else:
                plt.show()
            plt.close()

    if args.expt_name is not None:
        outfile.close()

    plt.figure()
    compare_plots_with_ground_truth(
        (optimizer.target, inner_lrs),
        fenics_functions,
        ground_truth_source,
        ground_truth_bc,
        ground_truth_geo,
    )
    save_opt(optimizer)


    if args.expt_name is not None:
        plt.savefig(os.path.join(args.out_dir, args.expt_name + "_viz_final.png"))
    else:
        plt.show()
    plt.close()
