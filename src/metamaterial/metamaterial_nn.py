import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from ..nets.field import NeuralField

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
parser.add_argument("--outer_lr", type=float, default=3e-5, help="outer learning rate")
parser.add_argument(
    "--boundary_points",
    type=int,
    default=2048,
    help="num points on the boundary for inner loss",
)
parser.add_argument(
    "--domain_points",
    type=int,
    default=2048,
    help="num points inside the domain for inner loss",
)
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
    "--bc_scale", type=float, default=0.1, help="scale of bc displacement"
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
parser.add_argument("--out_dir", type=str, default="mm_results")
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


    @partial(jax.jit, static_argnums=(4, 5, 6))
    def train_step(
        points_in_domain,
        points_on_boundary,
        points_on_interior_boundary,
        optimizer,
        source_params,
        bc_params,
        interior_weight,
    ):
        boundary_loss, boundary_grad = jax.value_and_grad(
            lambda model: args.bc_weight
            * boundary_loss_fn(points_on_boundary, model, bc_params)
        )(optimizer.target)


        domain_loss, domain_grad = jax.value_and_grad(
            lambda model: domain_loss_fn(points_in_domain, model, source_params)
        )(optimizer.target)


        interior_boundary_loss, interior_boundary_grad = jax.value_and_grad(
            lambda model: args.interior_weight
            * interior_bc_loss_fn(
                points_on_interior_boundary, model, geo_params, source_params
            )
        )(optimizer.target)

        boundary_grad_norm = np.sqrt(
            np.sum([(x ** 2).sum() for x in jax.tree_util.tree_flatten(boundary_grad)[0]])
        )
        domain_grad_norm = np.sqrt(
            np.sum([(x ** 2).sum() for x in jax.tree_util.tree_flatten(domain_grad)[0]])
        )
        interior_boundary_grad_norm = np.sqrt(
            np.sum([(x ** 2).sum() for x in jax.tree_util.tree_flatten(
                interior_boundary_grad)[0]])
        )

        if args.pcgrad > 0.0:
            project = partial(pcgrad.project_grads, args.pcgrad)
            domain_grad_ = jax.tree_multimap(project, domain_grad, boundary_grad,
                                             interior_boundary_grad)
            interior_boundary_grad_ = jax.tree_multimap(project, interior_boundary_grad,
                                                        domain_grad,
                                                        boundary_grad)
            boundary_grad_ = jax.tree_multimap(project, boundary_grad,
                                                        interior_boundary_grad,
                                                        domain_grad)
            domain_grad, interior_boundary_grad, boundary_grad = (
                domain_grad_, interior_boundary_grad_, boundary_grad_
            )

        bgrad_flat, treedef = jax.tree_flatten(boundary_grad)
        dgrad_flat, treedef = jax.tree_flatten(domain_grad)
        igflat, treedef = jax.tree_flatten(interior_boundary_grad)
        total_grad = jax.tree_unflatten(
            treedef, [g1 + g2 + g3 for g1, g2, g3 in zip(bgrad_flat, dgrad_flat, igflat)]
        )

        total_loss = boundary_loss + interior_boundary_loss + domain_loss
        optimizer = optimizer.apply_gradient(total_grad)
        grad_flat, _ = jax.tree_flatten(total_grad.params)
        total_grad_norm = np.sqrt(np.sum([(g**2).sum() for g in grad_flat]))

        return (optimizer, total_loss, boundary_loss, domain_loss, interior_boundary_loss,
                boundary_grad_norm, domain_grad_norm, interior_boundary_grad_norm)


    Field = NeuralField.partial(
        sizes=[args.layer_size for _ in range(args.num_layers)],
        dense_args=(),
        nonlinearity=np.sin if args.siren else nn.swish,
        n_fourier=args.n_fourier,
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    _, init_params = Field.init_by_shape(subkey, [((1, 2), DTYPE)])
    model = flax.nn.Model(Field, init_params)

    optimizer = flax.optim.Adam(learning_rate=args.outer_lr).create(model)

    with Timer() as t:
        key, subkey = jax.random.split(key)
        source_params, bc_params, geo_params = sample_params(subkey, args)
    log("made params in {}s".format(t.interval))

    log("source params: ", source_params)
    log("bc params: ", bc_params)
    log("geo params: ", geo_params)

    with Timer() as t:
        ground_truth = solve_fenics(source_params, bc_params, geo_params)
        ground_truth.set_allow_extrapolation(True)
    log("made ground truth in {}s".format(t.interval))

    plt.figure()
    plt.subplot(3, 1, 1)
    opt_fenics = make_fenics(optimizer.target, geo_params)
    fa.plot(opt_fenics, mode="displacement", title="neural")
    plt.subplot(3, 1, 2)
    fa.plot(ground_truth, mode="displacement", title="truth")
    plt.subplot(3, 1, 3)
    diff = fa.project(opt_fenics - ground_truth, ground_truth.function_space())
    fa.plot(
        diff, mode="displacement", title="difference",
    )
    plt.savefig("mm_init.png")

    key, k1, k2, k3 = jax.random.split(key, 4)
    points_in_domain_test = sample_points_in_domain(k1, args.domain_points, geo_params)
    points_on_boundary_test = sample_points_on_boundary(
        k2, args.boundary_points, geo_params
    )
    points_on_interior_boundary_test = sample_points_on_interior_boundary(
        k3, args.boundary_points, geo_params
    )
    plt.figure()
    plt.scatter(points_in_domain_test[:, 0], points_in_domain_test[:, 1], label="domain")
    plt.scatter(
        points_on_boundary_test[:, 0], points_on_boundary_test[:, 1], label="boundary"
    )
    plt.scatter(
        points_on_interior_boundary_test[:, 0],
        points_on_interior_boundary_test[:, 1],
        label="inner_boundary",
    )
    plt.legend()
    plt.savefig("mm_sampling.png")

    true = np.array([ground_truth(point) for point in points_in_domain_test]).astype(
        DTYPE)

    for step in range(args.outer_steps):
        key, sk1, sk2, sk3 = jax.random.split(key, 4)

        points_in_domain = sample_points_in_domain(sk1, args.domain_points, geo_params)
        points_on_boundary = sample_points_on_boundary(
            sk2, args.boundary_points, geo_params
        )
        points_on_interior_boundary = sample_points_on_interior_boundary(
            sk3, args.boundary_points, geo_params
        )
        (optimizer, loss, loss_bc, loss_dom, loss_int,
         grad_bc, grad_dom, grad_int) = train_step(
            points_in_domain,
            points_on_boundary,
            points_on_interior_boundary,
            optimizer,
            source_params,
            bc_params,
            args.interior_weight,
        )
        preds = optimizer.target(points_in_domain_test)

        try:
            true_ = true.reshape(preds.shape).astype(DTYPE)
            supervised_rmse = np.sqrt(np.mean((preds - true_) ** 2))
        except Exception as e:
            pdb.set_trace()
        log(
            "step {}, loss {}, loss_boundary {}, loss_domain {}, loss_interior {}, "
            "grad_bc {}, grad_dom {}, grad_interior {}, supervised_err {}".format(
                step,
                float(loss),
                float(loss_bc),
                float(loss_dom),
                float(loss_int),
                float(grad_bc),
                float(grad_dom),
                float(grad_int),
                supervised_rmse,
            ),
        )

        if args.viz_every > 0 and step % args.viz_every == 0:
            plt.figure()
            plt.subplot(3, 1, 1)
            fa.plot(make_fenics(optimizer.target, geo_params), mode="displacement", title="neural")
            plt.title('pred')
            plt.subplot(3, 1, 2)
            fa.plot(ground_truth, mode="displacement", title="truth")
            plt.title('true')
            plt.subplot(3, 1, 3)
            fa.plot(
                make_fenics(
                    lambda xs: optimizer.target(xs).reshape(-1, 2)
                    - np.array([ground_truth(*x) for x in xs]).reshape(-1, 2),
                    geo_params,
                ),
                mode="displacement",
                title="difference",
            )
            plt.title('pred-true')
            if args.expt_name is not None:
                plt.savefig(os.path.join(args.out_dir, expt_name +
                                         "_viz_step_{}.png".format(step)))
            else:
                plt.show()


    if args.expt_name is not None:
        outfile.close()

    plt.figure()
    plt.subplot(3, 1, 1)
    fa.plot(make_fenics(optimizer.target, geo_params),
            mode="displacement", title="neural")
    plt.title('pred')
    plt.subplot(3, 1, 2)
    fa.plot(ground_truth, mode="displacement", title="truth")
    plt.title('true')
    plt.subplot(3, 1, 3)
    fa.plot(
        make_fenics(
            lambda xs: optimizer.target(xs).reshape(-1, 2).astype(np.float32)
            - np.array([ground_truth(*x) for x in xs]).reshape(-1, 2),
            geo_params,
        ),
        mode="displacement",
        title="difference",
    )
    plt.title('pred-true')
    if args.expt_name is not None:
        plt.savefig(os.path.join(args.out_dir, expt_name + "_viz_final.png"))
    else:
        plt.show()
