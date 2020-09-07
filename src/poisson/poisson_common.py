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
    err_on_boundary = vmap_boundary_conditions(points_on_boundary, bc_params).reshape(
        -1, 1
    ) - potential_fn(points_on_boundary).reshape(-1, 1)
    loss_on_boundary = np.mean(err_on_boundary ** 2)
    err_in_domain = vmap_laplace_operator(points_in_domain, potential_fn).reshape(
        -1, 1
    ) - vmap_source(points_in_domain, source_params).reshape(-1, 1)
    loss_in_domain = np.mean(err_in_domain ** 2)
    return loss_on_boundary, loss_in_domain


@partial(jax.jit, static_argnums=(1,))
def sample_params(key, args):
    k1, k2, k3 = jax.random.split(key, 3)
    if args.vary_source:
        source_params = jax.random.normal(k1, shape=(2, 3,))
    else:
        source_params = None
    if args.vary_bc:
        bc_params = jax.random.normal(k2, shape=(5,))
    else:
        bc_params = None
    if args.vary_geometry:
        geo_params = jax.random.uniform(k3, minval=-0.2, maxval=0.2, shape=(2,))
    else:
        geo_params = np.array([0.0]), np.array([0.0])

    return source_params, bc_params, geo_params


def sample_points_on_boundary(key, n, geo_params=None):
    if geo_params is not None:
        c1, c2 = geo_params
    else:
        c1, c2 = (0.0, 0.0)
    theta = np.linspace(0.0, 2 * np.pi, n)
    theta = theta + jax.random.uniform(
        key, minval=0.0, maxval=(2 * np.pi / n), shape=(n,)
    )
    r0 = 1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)
    x = r0 * np.cos(theta)
    y = r0 * np.sin(theta)
    return np.stack([x, y], axis=1)


def sample_points_in_domain(key, n, geo_params=None):
    if geo_params is not None:
        c1, c2 = geo_params
    else:
        c1, c2 = (0.0, 0.0)
    key1, key2, key3 = jax.random.split(key, 3)
    theta = np.linspace(0.0, 2 * np.pi, n)
    theta = theta + jax.random.uniform(
        key1, minval=0.0, maxval=(2 * np.pi / n), shape=(n,)
    )
    r0 = 1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)
    dr = np.linspace(0.0, 1.0 - 1.0 / n, n)
    dr = jax.random.shuffle(key2, dr)
    dr = dr + jax.random.uniform(key3, minval=0.0, maxval=1.0 / n, shape=(n,))
    r = dr * r0
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1)


def boundary_conditions(r, x):
    """
    This returns the value required by the dirichlet boundary condition at x.
    """
    if r is None:
        return 0.0
    else:
        theta = np.arctan2(x[1], x[0])
        return (
            r[0]
            + r[1] / 4 * np.cos(theta)
            + r[2] / 4 * np.sin(theta)
            + r[3] / 4 * np.cos(2 * theta)
            + r[4] / 4 * np.sin(2 * theta)
        )


def vmap_boundary_conditions(points_on_boundary, bc_params):
    return vmap(partial(boundary_conditions, bc_params))(points_on_boundary)


def source(r, x):
    if r is None:
        return np.array([1.0])
    else:
        result = np.array([0.0])
        for n in range(r.shape[0]):
            result += (
                r[n, 2] * 1e2 * np.exp(-((x[0] - r[n, 0]) ** 2 + (x[1] - r[n, 1]) ** 2))
            )
        return result


def vmap_source(points_in_domain, source_params):
    return vmap(partial(source, source_params))(points_in_domain)
