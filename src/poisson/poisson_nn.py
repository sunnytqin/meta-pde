import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from ..nets.field import NeuralPotential

from functools import partial
import flax
from flax import nn

from .poisson_fenics import solve_fenics
from .poisson_common import *
from ..util import pcgrad

import matplotlib.pyplot as plt
import pdb
import sys
import os

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--outer_lr", type=float, default=3e-3, help="outer learning rate")
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
parser.add_argument("--outer_steps", type=int, default=int(1e6), help="num outer steps")
parser.add_argument("--num_layers", type=int, default=5, help="num fcnn layers")
parser.add_argument("--layer_size", type=int, default=256, help="fcnn layer size")
parser.add_argument("--vary_source", type=int, default=1, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=1, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=1, help="1=true.")
parser.add_argument("--siren", type=int, default=0, help="1=true.")
parser.add_argument("--pcgrad", type=float, default=0.0, help="1=true.")
parser.add_argument("--bc_weight", type=float, default=1e1, help="weight on bc loss")
parser.add_argument("--out_dir", type=str, default="poisson_results")
parser.add_argument("--expt_name", type=str, default=None)
parser.add_argument("--viz_every", type=int, default=int(1e3),
                    help="plot every N steps")



loss_fn = jax.jit(loss_fn)


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


    Field = NeuralPotential.partial(
        sizes=[args.layer_size for _ in range(args.num_layers)],
        dense_args=(),
        nonlinearity=np.sin if args.siren else nn.swish,
    )

    key, subkey = jax.random.split(jax.random.PRNGKey(0))

    _, init_params = Field.init_by_shape(subkey, [((1, 2), DTYPE)])
    model = flax.nn.Model(Field, init_params)

    optimizer = flax.optim.Adam(learning_rate=args.outer_lr).create(model)


    grid = npo.zeros([101, 101, 2])
    for i in range(101):
        for j in range(101):
            grid[i, j] += [i * 1.0 / 100, j * 1.0 / 100]
    grid = np.array(grid).astype(DTYPE).reshape(-1, 2) * 2 - np.array(
        [[1.0, 1.0]]).astype(DTYPE)


    @jax.jit
    def train_step(
        points_in_domain, points_on_boundary, optimizer, source_params, bc_params
    ):
        loss_on_batch = lambda model: loss_fn(
            points_on_boundary, points_in_domain, model, source_params, bc_params
        )

        loss_bc, grad_bc = jax.value_and_grad(
            lambda model: loss_on_batch(model)[0] * args.bc_weight
        )(optimizer.target)
        loss_dom, grad_dom = jax.value_and_grad(lambda model: loss_on_batch(model)[1])(
            optimizer.target
        )
        grad_bc_norm = np.sqrt(
            np.sum([(x ** 2).sum() for x in jax.tree_util.tree_flatten(grad_bc)[0]])
        )
        grad_dom_norm = np.sqrt(
            np.sum([(x ** 2).sum() for x in jax.tree_util.tree_flatten(grad_dom)[0]])
        )
        if args.pcgrad > 0.0:
            project = partial(pcgrad.project_grads, args.pcgrad)
            grad_bc_ = jax.tree_multimap(project, grad_bc, grad_dom)
            grad_dom_ = jax.tree_multimap(project, grad_dom, grad_bc)
            grad_bc, grad_dom = grad_bc_, grad_dom_

        loss = loss_bc + loss_dom
        gradient = jax.tree_multimap(lambda x, y: x + y, grad_bc, grad_dom)


        optimizer = optimizer.apply_gradient(gradient)
        return (optimizer, loss, loss_bc, loss_dom, grad_bc_norm, grad_dom_norm)


    key, subkey = jax.random.split(key)

    source_params, bc_params, geo_params = sample_params(subkey, args)

    log("source params: ", source_params)
    log("bc params: ", bc_params)
    log("geo params: ", geo_params)

    ground_truth = solve_fenics(source_params, bc_params, geo_params)
    ground_truth.set_allow_extrapolation(True)

    points_in_domain_test = sample_points_in_domain(
        jax.random.PRNGKey(3), args.domain_points, geo_params
    )

    true = np.array(
        [ground_truth(point) for point in points_in_domain_test]
    )

    for step in range(args.outer_steps):
        key, sk1, sk2 = jax.random.split(key, 3)

        points_in_domain = sample_points_in_domain(sk1, args.domain_points, geo_params)
        points_on_boundary = sample_points_on_boundary(
            sk2, args.boundary_points, geo_params
        )
        optimizer, loss, loss_bc, loss_dom, grad_bc, grad_dom = train_step(
            points_in_domain, points_on_boundary, optimizer, source_params, bc_params
        )

        preds = optimizer.target(points_in_domain_test)
        try:
            true_ = true.reshape(preds.shape)
            supervised_rmse = np.sqrt(np.mean((preds - true_) ** 2))
            # pdb.set_trace()
        except Exception as e:
            pdb.set_trace()
        log(
            "step {}, loss {}, loss_boundary {}, loss_domain {}, "
            "grad_bc {}, grad_dom {}, supervised_err {}".format(
                step,
                float(loss),
                float(loss_bc),
                float(loss_dom),
                float(grad_bc),
                float(grad_dom),
                supervised_rmse,
            ),
        )

        if args.viz_every > 0 and step % args.viz_every == 0:
            plt.figure()
            plt.subplot(3, 1, 1)
            plot(optimizer.target, grid, source_params, bc_params, geo_params)
            plt.subplot(3, 1, 2)
            plot(
                lambda xs: np.array([ground_truth(x) for x in xs]),
                grid,
                source_params,
                bc_params,
                geo_params,
            )
            plt.subplot(3, 1, 3)
            plot(
                lambda xs: optimizer.target(xs).reshape(-1, 1)
                - np.array([ground_truth(x) for x in xs]).reshape(-1, 1),
                grid,
                source_params,
                bc_params,
                geo_params,
            )
            if args.expt_name is not None:
                plt.savefig(os.path.join(
                    args.out_dir, args.expt_name +
                    "_viz_step_{}.png".format(step)))
            else:
                plt.show()

    if args.expt_name is not None:
        outfile.close()

    plt.figure()
    plt.subplot(3, 1, 1)
    plot(optimizer.target, grid, source_params, bc_params, geo_params)
    plt.title('pred')
    plt.subplot(3, 1, 2)
    plot(
        lambda xs: np.array([ground_truth(x) for x in xs]),
        grid,
        source_params,
        bc_params,
        geo_params,
    )
    plt.title('truth')
    plt.subplot(3, 1, 3)
    plot(
        lambda xs: optimizer.target(xs).reshape(-1, 1).astype(np.float32)
        - np.array([ground_truth(x) for x in xs]).reshape(-1, 1),
        grid,
        source_params,
        bc_params,
        geo_params,
    )
    plt.title('pred-true')
    if args.expt_name is not None:
        plt.savefig(os.path.join(args.out_dir, args.expt_name + "_viz_final.png"))
    else:
        plt.show()
