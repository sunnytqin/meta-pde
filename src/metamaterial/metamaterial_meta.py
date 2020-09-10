import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from ..nets.gradient_conditioned import GradientConditionedField

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

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--bsize", type=int, default=2, help="batch size (in tasks)")
parser.add_argument("--n_eval", type=int, default=2, help="num eval tasks")
parser.add_argument("--inner_lr", type=float, default=1e-5, help="inner learning rate")
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
parser.add_argument("--inner_steps", type=int, default=2, help="num inner steps")
parser.add_argument("--outer_steps", type=int, default=int(1e4), help="num outer steps")
parser.add_argument("--num_layers", type=int, default=5, help="num fcnn layers")
parser.add_argument("--n_fourier", type=int, default=None, help="num fourier features")
parser.add_argument("--layer_size", type=int, default=256, help="fcnn layer size")
parser.add_argument("--siren", type=int, default=0, help="1 for true")
parser.add_argument("--pcgrad", type=float, default=0.0, help="1=true.")
parser.add_argument("--vary_source", type=int, default=1, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=1, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=1, help="1=true.")
parser.add_argument(
    "--bc_scale", type=float, default=0.05, help="scale of bc displacement"
)
parser.add_argument(
    "--interior_weight",
    type=float,
    default=1.0,
    help="weight on interior boundary loss",
)
parser.add_argument(
    "--bc_weight",
    type=float,
    default=1.0,
    help="weight on outer boundary loss",
)
parser.add_argument("--out_dir", type=str, default="mm_meta_results")
parser.add_argument("--expt_name", type=str, default=None)
parser.add_argument("--viz_every", type=int, default=0, help="plot every N steps")


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



    def loss_fn(field_fn, points_on_boundary, points_in_domain,
                points_on_interior, source_params, bc_params, geo_params):

        boundary_loss = boundary_loss_fn(points_on_boundary, field_fn, bc_params)
        domain_loss = domain_loss_fn(points_in_domain, field_fn, source_params)
        interior_loss = interior_bc_loss_fn(
            points_on_interior, field_fn, geo_params, source_params)

        return (
            args.bc_weight * boundary_loss +
            domain_loss +
            args.interior_weight * interior_loss)


    def get_meta_loss(
        points_in_domain,
        points_on_boundary,
        points_on_interior,
        inner_in_domain,
        inner_on_boundary,
        inner_on_interior,
        source_params,
        bc_params,
        geo_params
    ):
        field_fn = lambda model: partial(
            model,
            inner_loss_kwargs={
                "points_in_domain": inner_in_domain,
                "points_on_boundary": inner_on_boundary,
                "points_on_interior": inner_on_interior,
                "source_params": source_params,
                "bc_params": bc_params,
                "geo_params": geo_params
            },
        )

        def meta_loss(model):
            return loss_fn(
                points_on_boundary=points_on_boundary,
                points_in_domain=points_in_domain,
                points_on_interior=points_on_interior,
                source_params=source_params,
                bc_params=bc_params,
                geo_params=geo_params,
                field_fn=field_fn(model),
            )

        return meta_loss

    def get_single_example_loss(args, key):
        # The input key is terminal
        k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
        source_params, bc_params, geo_params = sample_params(k1, args)
        points_in_domain = sample_points_in_domain(k2, args.outer_points, geo_params)
        points_on_boundary = sample_points_on_boundary(k3, args.outer_points, geo_params)
        points_on_interior = sample_points_on_interior_boundary(
            k4, args.outer_points, geo_params
        )
        inner_in_domain = sample_points_in_domain(k5, args.inner_points, geo_params)
        inner_on_boundary = sample_points_on_boundary(k6, args.inner_points, geo_params)
        inner_on_interior = sample_points_on_interior_boundary(
            k7, args.inner_points, geo_params
        )
        return get_meta_loss(
            points_in_domain=points_in_domain,
            points_on_boundary=points_on_boundary,
            points_on_interior=points_on_interior,
            inner_in_domain=inner_in_domain,
            inner_on_boundary=inner_on_boundary,
            inner_on_interior=inner_on_interior,
            source_params=source_params,
            bc_params=bc_params,
            geo_params=geo_params,
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
        true_fields = []
        coords = []
        for sp, bp, gp in zip(source_params_list, bc_params_list, geo_params_list):
            c1, c2 = gp
            ground_truth = solve_fenics(sp, bp, gp)
            fenics_functions.append(ground_truth)
            true_fields.append(np.array(ground_truth.vector()[:])).astype(DTYPE)
            coords.append(
                np.array(ground_truth.function_space().tabulate_dof_coordinates())[::2]
            ).astype(DTYPE)
        return fenics_functions, true_fields, coords


    def make_field_func(key, optimizer,
                        source_params, bc_params, geo_params, coords, args):
        # Input key is terminal
        k1, k2, k3 = jax.random.split(key, 3)
        inner_in_domain = sample_points_in_domain(k1, args.inner_points, geo_params)
        inner_on_boundary = sample_points_on_boundary(k2, args.inner_points, geo_params)
        inner_on_interior = sample_points_on_interior_boundary(
            k3, args.inner_points, geo_params)
        field_fn = partial(
            optimizer.target,
            inner_loss_kwargs={
                "points_in_domain": inner_in_domain,
                "points_on_boundary": inner_on_boundary,
                "points_on_interior": inner_on_interior,
                "source_params": source_params,
                "bc_params": bc_params,
                "geo_params": geo_params,
            },
        )
        field_vals = np.squeeze(field_fn(coords))

        # We have something of shape [n_dofs, 2] = Aij, i=1..n_dovs, j=1,2
        # We want a corresponding vector of scalars, shape [2*n_dofs]
        # But Fenics represents this vector as
        #  [A00, A01, A10, A11, A20, A21 ...], whereas calling np.reshape(A, -1)
        # gives [A00, A10, A20, ..., A01, A11, A21 ...]
        # So take the transpose before flattening to get the vector corresponding
        # to the Fenics representation
        field_vals = np.transpose(field_vals).reshape(-1)
        return field_vals


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
                field_vals = make_field_func(
                    keys[i],
                    optimizer,
                    sources[i],
                    bc_params[i],
                    geo_params[i],
                    ground_truth.function_space().tabulate_dof_coordinates()[::2],
                    args,
                ).astype(np.float32)

                u_approx = fa.Function(ground_truth.function_space())
                assert len(field_vals) == len(np.array(u_approx.vector()[:]))
                assert len(np.array(u_approx.vector()[:]).shape) == 1

                u_approx.vector().set_local(field_vals)

                plt.subplot(4, N // 2, 1 + i)

                fa.plot(u_approx, title="Approx", mode='displacement')
                # bottom two rots are ground truth
                plt.subplot(4, N // 2, 1 + N + i)
                fa.plot(ground_truth, title="Truth", mode='displacement')


    def vmap_validation_error(
        optimizer,
        ground_truth_source,
        ground_truth_bc,
        ground_truth_geo,
        trunc_coords,
        trunc_true_fields,
        args,
    ):
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, args.n_eval)
        fields = vmap(make_field_func, (0, None, 0, 0, 0, 0, None))(
            keys,
            optimizer,
            ground_truth_source,
            ground_truth_bc,
            ground_truth_geo,
            trunc_coords,
            args,
        )

        return np.mean((fields - trunc_true_fields) ** 2)


    GCField = GradientConditionedField.partial(
        inner_steps=args.inner_steps,
        inner_lr=args.inner_lr,
        train_inner_lrs=True,
        inner_loss=loss_fn,
        # n_fourier=5,
        base_args={
            "sizes": [args.layer_size for _ in range(args.num_layers)],
            "input_dim": 2,
            "output_dim": 2,
            "nonlinearity": np.sin if args.siren else nn.swish,
            "kernel_init": flax.nn.initializers.variance_scaling(
                2.0, "fan_in", "truncated_normal"
            ),
        },  # These base_args are the arguments for the initialization of GradientConditionedFieldParameters
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    _, init_params = GCField.partial(
        inner_loss_kwargs={}, inner_loss=lambda x: 0.0
    ).init_by_shape(subkey, [((1, 2), DTYPE)])

    model = flax.nn.Model(GCField, init_params)

    optimizer = flax.optim.Adam(learning_rate=args.outer_lr).create(model)

    assert args.n_eval % 2 == 0

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
            optimizer, loss = batch_train_step(args, optimizer, subkey)
            loss = float(loss)

        val_error = vmap_validation_error(
            optimizer,
            ground_truth_source,
            ground_truth_bc,
            ground_truth_geo,
            trunc_coords,
            trunc_true_fields,
            args,
        )
        log("step: {}, loss: {}, val: {}, time: {}".format(
                step, loss, val_error, t.interval))

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
                plt.savefig(os.path.join(args.out_dir, expt_name +
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
