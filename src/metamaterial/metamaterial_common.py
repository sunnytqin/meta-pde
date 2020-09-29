import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from functools import partial
import flax
from flax import nn
from ..util.timer import Timer


DTYPE = np.float32


def divergence(u, x):
    dudx = jax.jacfwd(u)(x)
    return np.trace(dudx)


def tensor_divergence(u, x):
    # u(x) is d x d
    dudx = jax.jacfwd(u)(x)  # d x d x d
    return np.concatenate([dudx[:, 0, 0], dudx[:, 1, 1]]).transpose()


def interior_bc_loss_fn(points_on_interior_boundary, u, geo_params, source_params):
    # pdb.set_trace()
    lossval = vmap(partial(interior_bc, geo_params, source_params, u))(
        points_on_interior_boundary
    ).mean()
    return lossval


def boundary_loss_fn(points_on_boundary, field_fn, bc_params):
    err_on_boundary = vmap_boundary_conditions(points_on_boundary, bc_params).reshape(
        -1, 2
    ) - field_fn(points_on_boundary).reshape(-1, 2)
    loss_on_boundary = (err_on_boundary ** 2).mean()
    return loss_on_boundary


def domain_loss_fn(points_in_domain, field_fn, source_params):
    loss_in_domain = np.sum(
        vmap_nhe_violation(points_in_domain, field_fn, source_params)
    )
    # loss_in_domain = (err_in_domain ** 2).mean()
    return loss_in_domain


# Deformation gradient
def defgrad(u):
    """
    Inputs:
        u: fn mapping [2]  -> [2]
    Outputs:
        F: fn mapping [2] -> [2, 2]
    """

    def _dg(x):
        x = x.reshape(2)
        dudx = jax.jacfwd(lambda x: u(x).reshape(2))(x)
        # pdb.set_trace()
        assert dudx.shape[0] == 2 and dudx.shape[1] == 2
        return dudx + np.eye(2)

    return _dg


def neo_hookean_energy(u, source_params, x):
    """
    Inputs:
        u: fn mapping [n, 2] or [2] -> [n, 2] or [2]
        young_mod: scalar
        poisson_ratio: scalar

    Outputs:
        energy: [n, 1] or [1]
    """
    young_mod, poisson_ratio = source_params
    # if poisson_ratio >= 0.5:
    #    raise ValueError(
    #        "Poisson's ratio must be below isotropic upper limit 0.5. Found {}".format(
    #            poisson_ratio
    #        )
    #    )
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))

    x = x.reshape(2)
    F = defgrad(u)(x)
    J = np.linalg.det(F)
    rcg = np.matmul(F.transpose(), F)  # right cauchy green
    Jinv = 1.0 / (J + 1e-14)
    I1 = np.trace(rcg)  # first invariant

    energy = (shear_mod / 2) * (Jinv * I1 - 2) + (bulk_mod / 2) * (J - 1) ** 2

    return energy


def nhe_given_F(F, source_params):
    young_mod, poisson_ratio = source_params
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))
    J = np.linalg.det(F)
    rcg = np.matmul(F.transpose(), F)  # right cauchy green
    Jinv = 1.0 / (J + 1e-14)
    I1 = np.trace(rcg)  # first invariant

    energy = (shear_mod / 2) * (Jinv * I1 - 2) + (bulk_mod / 2) * (J - 1) ** 2
    return energy


def first_pk_ad(u, source_params, x):
    young_mod, poisson_ratio = source_params
    # if poisson_ratio >= 0.5:
    #    raise ValueError(
    #        "Poisson's ratio must be below isotropic upper limit 0.5. Found {}".format(
    #            poisson_ratio
    #        )
    #    )
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))

    x = x.reshape(2)
    F = defgrad(u)(x)
    return jax.grad(nhe_given_F)(F, source_params)


def first_pk_div_sumsq(u, source_params, x):
    return np.sum(tensor_divergence(lambda x: first_pk_ad(u, source_params, x), x) ** 2)


def inv2d(A):
    assert len(A.shape) == 2 and A.shape[0] == 2 and A.shape[1] == 2
    return (
        1.0
        / (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
        * np.array([[A[1, 1], A[1, 0]], [A[0, 1], A[0, 0]]])
    )


def first_pk_stress(u, source_params, x):

    young_mod, poisson_ratio = source_params
    # if poisson_ratio >= 0.5:
    #    raise ValueError(
    #        "Poisson's ratio must be below isotropic upper limit 0.5. Found {}".format(
    #            poisson_ratio
    #        )
    #    )
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))

    # x = x.reshape(2)
    F = defgrad(u)(x)
    J = np.linalg.det(F)
    rcg = np.matmul(F.transpose(), F)  # right cauchy green
    Jinv = 1.0 / (J + 1e-14)
    I1 = np.trace(rcg)  # first invariant

    FinvT = inv2d(F).transpose()
    first_pk_stress = (
        Jinv * shear_mod * (F - (1 / 2) * I1 * FinvT) + J * bulk_mod * (J - 1) * FinvT
    )
    return first_pk_stress


def second_pk_stress(u, source_params, x):

    young_mod, poisson_ratio = source_params
    # if poisson_ratio >= 0.5:
    #    raise ValueError(
    #        "Poisson's ratio must be below isotropic upper limit 0.5. Found {}".format(
    #            poisson_ratio
    #        )
    #    )
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))

    x = x.reshape(2)
    F = defgrad(u)(x)
    J = np.linalg.det(F)
    rcg = np.matmul(F.transpose(), F)  # right cauchy green
    Jinv = 1.0 / (J + 1e-14)
    I1 = np.trace(rcg)  # first invariant

    Finv = inv2d(F)
    first_pk_stress = (
        Jinv * shear_mod * (F - (1 / 2) * I1 * Finv.transpose())
        + J * bulk_mod * (J - 1) * Finv.transpose()
    )
    return np.matmul(Finv, first_pk_stress)


def vmap_nhe(x, u, source_params):
    return vmap(partial(neo_hookean_energy, u, source_params))(x).reshape(-1, 1)


def vmap_nhe_violation(x, u, source_params):
    return vmap(partial(first_pk_div_sumsq, u, source_params))(x).reshape(-1, 1)


def interior_bc(geo_params, source_params, u, x):
    """This loss returns <dEnergy/dx, N> where N is the normal to the boundary.
    It is a Nietsche boundary condition.
    """
    c1, c2 = geo_params
    assert x.shape[0] == 2 and len(x.shape) == 1

    theta = np.arctan2(x[1], x[0])

    def get_xy(theta):
        rval = r(theta, c1, c2)
        return np.array([rval * np.cos(theta), rval * np.sin(theta)])

    tangent = jax.jacfwd(get_xy)(theta)

    tangent = tangent.reshape(2)

    # <[a, b], [-b, a]> = ab - ba = 0
    normal = np.array([-tangent[1], tangent[0]])
    normal = (normal / np.linalg.norm(normal)).reshape(2, 1)

    # Non-stationary dispacements, i.e. du_dx = 0
    pk = first_pk_ad(u, source_params, x)  # dForce / dx
    assert len(pk.shape) == 2 and pk.shape[0] == 2 and pk.shape[1] == 2

    pk_dot_n = np.matmul(pk, normal)

    return np.sum(pk_dot_n ** 2)


def sample_points_on_boundary(key, n, geo_params=None):
    if geo_params is not None:
        c1, c2 = geo_params
    else:
        c1, c2 = np.zeros(2, dtype=DTYPE)
    theta = np.linspace(0.0, 2 * np.pi, n, dtype=DTYPE)
    theta = theta + jax.random.uniform(
        key, minval=0.0, maxval=(2 * np.pi / n), shape=(n,), dtype=DTYPE
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
        c1, c2 = np.zeros(2, dtype=DTYPE)
    theta = np.linspace(0.0, 2 * np.pi, n, dtype=DTYPE)
    theta = theta + jax.random.uniform(
        key, minval=0.0, maxval=(2 * np.pi / n), shape=(n,), dtype=DTYPE
    )
    rs = r(theta, c1, c2)
    x = rs * np.cos(theta)
    y = rs * np.sin(theta)
    return np.stack([x, y], axis=1)


@partial(jax.jit, static_argnums=(1, 2, 3))
def sample_points_in_domain_rejection(key, n, gridsize, geo_params=None):
    if geo_params is not None:
        c1, c2 = geo_params
    else:
        c1, c2 = np.zeros(2, dtype=DTYPE)

    def sample_grid(key):
        # We generate a semi-randomized uniform grid
        # Initialize the grid
        xg = np.linspace(-1, 1, gridsize, endpoint=False, dtype=DTYPE)
        yg = np.linspace(-1, 1, gridsize, endpoint=False, dtype=DTYPE)
        # Cartesian product
        xys = np.transpose(np.array([np.tile(xg, len(yg)), np.repeat(yg, len(xg))]))

        k1, k2 = jax.random.split(key, 2)
        # Add some random, jittering by the size of the linspace interval
        xys = xys + jax.random.uniform(
            k1, shape=(len(xys), 2,), minval=0.0, maxval=xg[1] - xg[0], dtype=DTYPE
        )
        xys = jax.random.permutation(k2, xys)[:n]

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
        return (i + 1, new_xs, new_accepted, new_key)

    init_xs = np.nan * np.ones((n, 2), dtype=DTYPE)
    init_accepted = jax.lax.full_like(np.ones(n), False, np.bool_, (n,))
    xs = jax.lax.while_loop(cond_fn, body_fn, (0, init_xs, init_accepted, key))[1]

    return xs


sample_points_in_domain = sample_points_in_domain_rejection


def sample_params(key, args):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    if args.vary_bc:
        bc_params = args.bc_scale * jax.random.uniform(
            k1, shape=(2, 7,), minval=-1.0, maxval=1.0, dtype=DTYPE
        )
    else:
        bc_params = np.zeros([2, 5], dtype=DTYPE)
    if args.vary_geometry:
        geo_params = jax.random.uniform(
            k2, minval=-0.03, maxval=0.03, shape=(2,), dtype=DTYPE
        )
    else:
        geo_params = np.zeros(2, dtype=DTYPE)
    if args.vary_source:
        young_mod = jax.random.uniform(k3, minval=0.7, maxval=2.0, dtype=DTYPE)
        poisson_ratio = jax.random.uniform(k4, minval=0.4, maxval=0.49, dtype=DTYPE)
    else:
        young_mod = 1.0
        poisson_ratio = 0.49

    return np.array([young_mod, poisson_ratio]), bc_params, geo_params


def r(theta, c1, c2, porosity=0.5):
    r0 = np.sqrt(2 * porosity) / np.sqrt(np.pi * (2 + c1 ** 2 + c2 ** 2))
    return r0 * (1 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta))


def boundary_conditions(r, x):
    """
    This returns the value required by the dirichlet boundary condition at x.
    """

    if r is None:
        return np.array([0.0, 0.0], dtype=np.float16)
    else:
        # pdb.set_trace()
        theta = np.arctan2(x[1], x[0])
        return np.array(
            [
                r[0, 0]
                + r[0, 1] * np.cos(theta)
                + r[0, 2] * np.sin(theta)
                + r[0, 3] * np.cos(2 * theta) / 2
                + r[0, 4] * np.sin(2 * theta) / 2
                + r[0, 5] * np.cos(4 * theta) / 4
                + r[0, 6] * np.sin(4 * theta) / 4,
                r[1, 0]
                + r[1, 1] * np.cos(theta)
                + r[1, 2] * np.sin(theta)
                + r[1, 3] * np.cos(2 * theta) / 2
                + r[1, 4] * np.sin(2 * theta) / 2
                + r[1, 5] * np.cos(4 * theta) / 4
                + r[1, 6] * np.sin(4 * theta) / 4,
            ]
        )


def vmap_boundary_conditions(points_on_boundary, bc_params):
    return vmap(partial(boundary_conditions, bc_params))(points_on_boundary)
