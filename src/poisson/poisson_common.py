import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap
from ..nets.field import vmap_laplace_operator

from functools import partial
import flax
from flax import nn

import matplotlib.pyplot as plt
import pdb


DTYPE = np.float32


def plot(model, grid, source_params, bc_params, geo_params=(0.0, 0.0)):
    c1, c2 = geo_params
    potentials = model(grid)
    potentials = npo.array(potentials)
    thetas = np.arctan2(grid[:, 1], grid[:, 0])
    r0s = 1.0 + c1 * np.cos(4 * thetas) + c2 * np.cos(8 * thetas)
    potentials[npo.linalg.norm(grid, axis=1) > r0s] = 0.0
    potentials = potentials.reshape(101, 101)
    plt.imshow(potentials)
    plt.colorbar()


def loss_fn(
    points_on_boundary, points_in_domain, potential_fn, source_params, bc_params
):
    err_on_boundary = vmap_boundary_conditions(
        points_on_boundary, bc_params
    ) - potential_fn(points_on_boundary)
    loss_on_boundary = np.mean(err_on_boundary ** 2)
    err_in_domain = vmap_laplace_operator(points_in_domain, potential_fn) - vmap_source(
        points_in_domain, source_params
    )
    loss_in_domain = np.mean(err_in_domain ** 2)
    return loss_on_boundary, loss_in_domain


@partial(jax.jit, static_argnums=(1,))
def sample_params(key, args):
    k1, k2, k3 = jax.random.split(key, 3)
    if args.vary_source:
        source_params = jax.random.normal(k1, shape=(2, 3,), dtype=DTYPE)
    else:
        source_params = None
    if args.vary_bc:
        bc_params = jax.random.normal(k2, shape=(5,), dtype=DTYPE)
    else:
        bc_params = None
    if args.vary_geometry:
        geo_params = jax.random.uniform(
            k3, minval=-0.2, maxval=0.2, shape=(2,), dtype=DTYPE
        )
    else:
        geo_params = np.zeros(2, dtype=DTYPE)

    return source_params, bc_params, geo_params


@partial(jax.jit, static_argnums=(1,))
def sample_points_on_boundary(key, n, geo_params):
    c1, c2 = geo_params
    theta = np.linspace(0.0, 2 * np.pi, n, dtype=DTYPE)
    theta = theta + jax.random.uniform(
        key, minval=0.0, maxval=(2 * np.pi / n), shape=(n,), dtype=DTYPE
    )
    r0 = 1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)
    x = r0 * np.cos(theta)
    y = r0 * np.sin(theta)
    return np.stack([x, y], axis=1)


@partial(jax.jit, static_argnums=(1,))
def sample_points_in_domain(key, n, geo_params):
    c1, c2 = geo_params
    key1, key2, key3 = jax.random.split(key, 3)
    theta = np.linspace(0.0, 2 * np.pi, n, dtype=DTYPE)
    theta = theta + jax.random.uniform(
        key1, minval=0.0, maxval=(2 * np.pi / n), shape=(n,), dtype=DTYPE
    )
    r0 = 1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)
    dr = np.linspace(0.0, 1.0 - 1.0 / n, n, dtype=DTYPE)
    dr = jax.random.permutation(key2, dr)
    dr = dr + jax.random.uniform(
        key3, minval=0.0, maxval=1.0 / n, shape=(n,), dtype=DTYPE
    )
    r = dr * r0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


@jax.jit
def boundary_conditions(r, x):
    """
    This returns the value required by the dirichlet boundary condition at x.
    """
    theta = np.arctan2(x[1], x[0])
    return (
        r[0]
        + r[1] / 4 * np.cos(theta)
        + r[2] / 4 * np.sin(theta)
        + r[3] / 4 * np.cos(2 * theta)
        + r[4] / 4 * np.sin(2 * theta)
    ).sum()


@jax.jit
def vmap_boundary_conditions(points_on_boundary, bc_params):
    return vmap(partial(boundary_conditions, bc_params))(points_on_boundary)


@jax.jit
def source(r, x):
    x = x.reshape([1, -1]) * np.ones([r.shape[0], x.shape[0]])
    results = r[:, 2] * np.exp(-((x[:, 0] - r[:, 0]) ** 2 + (x[:, 1] - r[:, 1]) ** 2))
    return results.sum()


@jax.jit
def vmap_source(points_in_domain, source_params):
    return vmap(partial(source, source_params))(points_in_domain)
