
import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap


def sample_points_on_boundary(key, n, geo_params=None):
    if geo_params is not None:
        c1, c2 = geo_params
    else:
        c1, c2 = (0.0, 0.0)
    theta = np.linspace(0.0, 2 * np.pi, n)
    theta = theta + jax.random.uniform(
        key, minval=0.0, maxval=(2 * np.pi / n), shape=(n,)
    )
    theta_in_pm_pidiv4 = np.mod((theta + np.pi / 4), np.pi / 2) - np.pi / 4

    square_radius = 1.0 / np.cos(theta_in_pm_pidiv4)

    x = square_radius * np.cos(theta)
    y = square_radius * np.sin(theta)

    return np.stack([x, y], axis=1)


def sample_points_on_interior_boundary(key, n, geo_params=None):
    if geo_params is not None:
        c1, c2 = geo_params
    else:
        c1, c2 = (0.0, 0.0)
    theta = np.linspace(0.0, 2 * np.pi, n)
    theta = theta + jax.random.uniform(
        key, minval=0.0, maxval=(2 * np.pi / n), shape=(n,)
    )
    rs = r(theta, c1, c2)
    x = rs * np.cos(theta)
    y = rs * np.sin(theta)
    return np.stack([x, y], axis=1)


def sample_points_in_domain_rejection(key, n, geo_params=None):
    if geo_params is not None:
        c1, c2 = geo_params
    else:
        c1, c2 = (0.0, 0.0)

    def sample_grid(key):
        # We generate a semi-randomized uniform grid
        # Initialize the grid
        nsqrt = (np.ceil(np.sqrt(n))).astype(int)

        xg = np.linspace(-1, 1, nsqrt, endpoint=False)
        yg = np.linspace(-1, 1, nsqrt, endpoint=False)
        # Cartesian product
        xys = np.transpose(np.array([np.tile(xg, len(yg)), np.repeat(yg, len(xg))]))

        k1, k2 = jax.random.split(key, 2)
        # Add some random, jittering by the size of the linspace interval
        xys = xys + jax.random.uniform(k1, shape=(len(xys), 2,),
                                       minval=0., maxval=xg[1]-xg[0])
        xys = jax.random.shuffle(k2, xys)[:n]

        return xys

    def cond_fn(carry):
        i, xs, accepted, key = carry
        return (~accepted).any()

    def body_fn(carry):
        i, xs, accepted, key = carry

        new_key, subkey = jax.random.split(key)

        sampled_xs = sample_grid(subkey)

        new_xs = np.where(accepted.reshape(-1, 1), xs, sampled_xs)

        thetas = np.arctan2(new_xs[:, 1], new_xs[:, 0]).reshape(-1)
        rvals = np.linalg.norm(new_xs, axis=1).reshape(-1)
        pore_rs = r(thetas, c1, c2).reshape(-1)

        new_accepted = rvals > pore_rs
        return (i+1, new_xs, new_accepted, new_key)

    init_xs = np.nan * np.ones((n, 2))
    init_accepted = jax.lax.full_like(np.ones(n), False, np.bool_, (n,))
    xs = jax.lax.while_loop(cond_fn, body_fn, (0, init_xs, init_accepted, key))[1]

    return xs



sample_points_in_domain = sample_points_in_domain_rejection



def sample_params(key, args):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    if args.vary_bc:
        bc_params = args.bc_scale * jax.random.normal(k1, shape=(2, 5,))
    else:
        bc_params = None
    if args.vary_geometry:
        geo_params = jax.random.uniform(k2, minval=-0.2, maxval=0.2, shape=(2,))
    else:
        geo_params = np.array([0.0]), np.array([0.0])
    if args.vary_source:
        young_mod = jax.random.uniform(k3, minval=0.7, maxval=2.0)
        poisson_ratio = jax.random.uniform(k4, minval=0.4, maxval=0.49)
    else:
        young_mod = 1.0
        poisson_ratio = 0.49

    return (young_mod, poisson_ratio), bc_params, geo_params


def r(theta, c1, c2, porosity=0.5):
    r0 = np.sqrt(2 * porosity) / np.sqrt(np.pi * (2 + c1 ** 2 + c2 ** 2))
    return r0 * (1 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta))



def boundary_conditions(r, x):
    """
    This returns the value required by the dirichlet boundary condition at x.
    """

    if r is None:
        return np.array([0.0, 0.0])
    else:
        # pdb.set_trace()
        theta = np.arctan2(x[1], x[0])
        return np.array(
            [
                r[0, 0]
                + r[0, 1] * np.cos(theta)
                + r[0, 2] * np.sin(theta)
                + r[0, 3] * np.cos(2 * theta)
                + r[0, 4] * np.sin(2 * theta),
                r[1, 0]
                + r[1, 1] * np.cos(theta)
                + r[1, 2] * np.sin(theta)
                + r[1, 3] * np.cos(2 * theta)
                + r[1, 4] * np.sin(2 * theta),
            ]
        )



def vmap_boundary_conditions(points_on_boundary, bc_params):
    return vmap(partial(boundary_conditions, bc_params))(points_on_boundary)
