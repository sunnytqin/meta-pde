import jax
import jax.numpy as np
import numpy as npo
import pdb
from functools import partial
import argparse
from collections import namedtuple

from jax.config import config
# config.update('jax_disable_jit', True)

import matplotlib.pyplot as plt

import fenics as fa

from ..nets.field import (
    vmap_divergence,
    vmap_divergence_tensor,
    divergence,
    divergence_tensor,
)
from ..util.jax_tools import tree_unstack

from absl import app
from absl import flags
from ..util import common_flags


flags.DEFINE_float("max_reynolds", 1e3, "Reynolds number scale")


FLAGS = flags.FLAGS


def loss_domain_fn(field_fn, points_in_domain, params):
    # u ux + v uy - Re(uxx + uyy) = 0
    # u vx + v vy - Re(vxx + vyy) = 0
    source_params, bc_params, per_hole_params, n_holes = params

    jac_fn = jax.jacfwd(field_fn)

    def lhs_fn(x):
        return np.matmul(jac_fn(x).reshape(2,2), field_fn(x).reshape(2, 1)).reshape(2)

    def rhs_fn(x):
        uxx = jax.jvp(lambda xi: jax.jvp(field_fn, (xi,), (np.array([1., 0.]),))[1],
                (x,), (np.array([1., 0.]),))[1]
        uyy = jax.jvp(lambda xi: jax.jvp(field_fn, (xi,), (np.array([0., 1.]),))[1],
                        (x,), (np.array([0., 1.]),))[1]
        return (1./ source_params[0]) * (uxx + uyy).reshape(2)

    return (jax.vmap(lhs_fn)(points_in_domain) - jax.vmap(rhs_fn)(points_in_domain))**2


def loss_inlet_fn(field_fn, points_on_inlet, params):
    source_params, bc_params, per_hole_params, n_holes = params
    sinusoidal_magnitude = np.sin(
        np.pi * (points_on_inlet[:, 1] - FLAGS.ymin) / (FLAGS.ymax - FLAGS.ymin)
    ).reshape(-1, 1)

    return (
        field_fn(points_on_inlet)
        - bc_params[0].reshape(1, 2) * sinusoidal_magnitude
    ) ** 2


def loss_outlet_fn(field_fn, points_on_outlet, params):
    source_params, bc_params, per_hole_params, n_holes = params
    sinusoidal_magnitude = np.sin(
        np.pi * (points_on_outlet[:, 1] - FLAGS.ymin) / (FLAGS.ymax - FLAGS.ymin)
    ).reshape(-1, 1)

    return (
        field_fn(points_on_outlet)
        - bc_params[1].reshape(1, 2) * sinusoidal_magnitude
    ) ** 2


def loss_noslip_fn(field_fn, points_noslip, params):
    source_params, bc_params, per_hole_params, n_holes = params
    return field_fn(points_noslip) ** 2


def loss_fn(field_fn, points, params):
    (
        points_on_inlet,
        points_on_outlet,
        points_on_walls,
        points_on_holes,
        points_in_domain,
    ) = points
    points_noslip = np.concatenate([points_on_walls, points_on_holes])

    return (
        {
            "loss_noslip": np.mean(loss_noslip_fn(field_fn, points_noslip, params)),
            "loss_inlet": np.mean(loss_inlet_fn(field_fn, points_on_inlet, params)),
            "loss_outlet": np.mean(loss_outlet_fn(field_fn, points_on_outlet, params)),
        },
        {
            "loss_domain": np.mean(loss_domain_fn(field_fn, points_in_domain, params)),
        },
    )


@jax.jit
def sample_params(key):
    if FLAGS.fixed_num_pdes is not None:
        key = jax.random.PRNGKey(
            jax.random.randint(
                key, (1,), np.array([0]), np.array([FLAGS.fixed_num_pdes])
            )[0]
        )

    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)

    # These keys will all be 0 if we're not varying that factor
    k1 = k1 * FLAGS.vary_source
    k2 = k2 * FLAGS.vary_bc
    k3 = k3 * FLAGS.vary_geometry
    k4 = k4 * FLAGS.vary_geometry
    k5 = k5 * FLAGS.vary_geometry
    k6 = k6 * FLAGS.vary_geometry
    k7 = k7 * FLAGS.vary_geometry


    source_params = FLAGS.max_reynolds * jax.random.uniform(k1, shape=(1,), minval=0., maxval=1.)

    bc_params = FLAGS.bc_scale * jax.random.uniform(
        k2, minval=-1.0, maxval=1.0, shape=(2,2,)
    )

    n_holes = jax.random.randint(
        k3, shape=(1,), minval=np.array([1]),
        maxval=np.array([FLAGS.max_holes + 1])
    )[0]

    pore_shapes = jax.random.uniform(
        k4, minval=-0.2, maxval=0.2, shape=(FLAGS.max_holes, 2,)
    )

    pore_sizes = jax.random.uniform(
        k6,
        minval=0.1,
        maxval=FLAGS.max_hole_size / n_holes,
        shape=(FLAGS.max_holes, 1),
    )

    min_step = FLAGS.max_hole_size

    xlow = FLAGS.xmin + 1.5 * FLAGS.max_hole_size
    xhigh = FLAGS.xmax - 1.5 * FLAGS.max_hole_size
    ylow = FLAGS.ymin + 1.5 * FLAGS.max_hole_size
    yhigh = FLAGS.ymax - 1.5 * FLAGS.max_hole_size

    pore_x0y0 = jax.random.uniform(k7,
                                   minval=np.array([[xlow, ylow]]),
                                   maxval=np.array([[xhigh, yhigh]]),
                                   shape=(FLAGS.max_holes, 2))

    validity = np.zeros(FLAGS.max_holes, dtype=np.int32)
    validity = validity.at[0].add(1)

    for j in range(1, FLAGS.max_holes):
        dists = np.sqrt(np.sum((pore_x0y0[j].reshape(1, 2) - pore_x0y0.reshape(-1, 2))**2,
                               axis=1, keepdims=True))
        space_needed = (pore_sizes[j].reshape(1, 1) + pore_sizes.reshape(-1, 1) + FLAGS.max_hole_size) * validity.reshape(-1, 1)
        is_valid = np.sum((dists - space_needed)<0) <= 0
        validity = validity.at[j].add(1*is_valid)
        # print(is_valid)

    # pdb.set_trace()

    permutation = np.argsort(validity)[::-1]

    per_hole_params = np.concatenate((pore_shapes, pore_x0y0, pore_sizes), axis=1)
    per_hole_params = per_hole_params[permutation]

    n_holes = np.minimum(n_holes, np.sum(validity))

    return source_params, bc_params, per_hole_params, n_holes


def is_in_hole(xy, pore_params, tol=1e-7):
    c1, c2, x0, y0, size = pore_params
    vector = xy - np.array([x0, y0])
    theta = np.arctan2(*vector)
    length = np.linalg.norm(vector)
    r0 = size * (1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta))
    return r0 > length + tol


@partial(jax.jit, static_argnums=(1,))
def sample_points(key, n, params):
    _, _, per_hole_params, n_holes = params
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    ratio = (FLAGS.xmax - FLAGS.xmin) / (FLAGS.ymax - FLAGS.ymin)
    n_on_inlet = n // 12
    n_on_outlet = n_on_inlet
    n_on_walls = n // 6
    n_on_holes = n // 2 - n_on_walls - n_on_inlet - n_on_outlet
    points_on_inlet = sample_points_on_inlet(k1, n_on_inlet, params)
    points_on_outlet = sample_points_on_outlet(k2, n_on_outlet, params)
    points_on_walls = sample_points_on_walls(k3, n_on_walls, params)
    points_on_holes = sample_points_on_pores(k4, n_on_holes, params)
    points_in_domain = sample_points_in_domain(k5, n, params)
    return (
        points_on_inlet,
        points_on_outlet,
        points_on_walls,
        points_on_holes,
        points_in_domain,
    )


@partial(jax.jit, static_argnums=(1,))
def sample_points_on_inlet(key, n, params):
    _, _, per_hole_params, n_holes = params
    lhs_y = np.linspace(FLAGS.ymin, FLAGS.ymax, n, endpoint=False) + jax.random.uniform(
        key, minval=0.0, maxval=(FLAGS.ymax - FLAGS.ymin) / n, shape=(1,)
    )
    lhs = np.stack([FLAGS.xmin * np.ones(n), lhs_y], axis=1)
    return lhs


def sample_points_on_outlet(key, n, params):
    return sample_points_on_inlet(key, n, params) + np.array(
        [[FLAGS.xmax - FLAGS.xmin, 0.0]]
    )


@partial(jax.jit, static_argnums=(1,))
def sample_points_on_walls(key, n, params):
    _, _, per_hole_params, n_holes = params
    k1, k2 = jax.random.split(key)
    top_x = np.linspace(
        FLAGS.xmin, FLAGS.xmax, n // 2, endpoint=False
    ) + jax.random.uniform(
        k1, minval=0.0, maxval=(FLAGS.xmax - FLAGS.xmin) / (n // 2), shape=(1,)
    )
    top = np.stack([top_x, FLAGS.ymax * np.ones(n // 2)], axis=1)

    bot_x = np.linspace(
        FLAGS.xmin, FLAGS.xmax, n - n // 2, endpoint=False
    ) + jax.random.uniform(
        k2, minval=0.0, maxval=(FLAGS.xmax - FLAGS.xmin) / (n - n // 2), shape=(1,)
    )
    bot = np.stack([bot_x, FLAGS.ymin * np.ones(n - n // 2)], axis=1)

    return np.concatenate([top, bot])


@partial(jax.jit, static_argnums=(1,))
def sample_points_on_pores(key, n, params):
    _, _, per_hole_params, n_holes = params
    keys = jax.random.split(key, 3)

    mask = np.arange(per_hole_params.shape[0], dtype=np.int32)
    mask = mask < n_holes

    total_sizes = np.sum(per_hole_params[:, -1] ** 2 * mask)

    dpore = np.linspace(0.0, total_sizes, n, endpoint=False)
    dpore = dpore + jax.random.uniform(
        keys[0], minval=0.0, maxval=(total_sizes / n), shape=(1,)
    )

    cum_sizes = np.cumsum(per_hole_params[:, -1] ** 2)

    # Number of pores already "filled"
    filled = dpore.reshape(-1, 1) > cum_sizes.reshape(1, -1)
    pore_idxs = np.int32(np.sum(filled, axis=1))

    remainder = dpore - np.sum(
        filled * per_hole_params[:, -1].reshape(1, -1) ** 2, axis=1
    )

    per_point_params = per_hole_params[pore_idxs]

    fraction = remainder / per_point_params[:, -1] ** 2
    theta = fraction * 2 * np.pi

    r0 = per_point_params[:, -1] * (
        1.0
        + per_point_params[:, 0] * np.cos(4 * theta)
        + per_point_params[:, 1] * np.cos(8 * theta)
    )
    x = r0 * np.cos(theta) + per_point_params[:, 2]
    y = r0 * np.sin(theta) + per_point_params[:, 3]

    hole_points = np.stack((x, y), axis=1)

    in_hole = jax.vmap(
        jax.vmap(is_in_hole, in_axes=(0, None)), in_axes=(None, 0), out_axes=1
    )(hole_points, per_hole_params)
    mask = np.arange(per_hole_params.shape[0], dtype=np.int32).reshape(1, -1)
    mask = mask < n_holes
    in_hole = in_hole * mask
    in_hole = np.any(in_hole, axis=1)

    valid = 1 - in_hole

    # Hack to sample valid at random until we have sampled all,
    # and then to start sampling the valid ones again, without touching
    # the others
    p = np.concatenate([valid * (1e-2 ** i) for i in range(FLAGS.max_holes * 2)])

    idxs = jax.random.choice(
        keys[4], n * FLAGS.max_holes * 2, replace=False, p=p, shape=(n,)
    )

    return np.tile(hole_points, (FLAGS.max_holes * 2, 1))[idxs]


@partial(jax.jit, static_argnums=(1,))
def sample_points_in_domain(key, n, params):
    _, _, per_hole_params, n_holes = params
    k1, k2 = jax.random.split(key)
    ratio = (FLAGS.xmax - FLAGS.xmin) / (FLAGS.ymax - FLAGS.ymin)
    # total number of points is 2 * n
    # so as long as the fraction of volume covered is << 1/2 we are ok
    n_x = npo.int32(npo.sqrt(2) * npo.sqrt(n) * npo.sqrt(ratio))
    n_y = npo.int32(npo.sqrt(2) * npo.sqrt(n) / npo.sqrt(ratio))
    dx = (FLAGS.xmax - FLAGS.xmin) / n_x
    dy = (FLAGS.ymax - FLAGS.ymin) / n_y

    xs = np.linspace(FLAGS.xmin, FLAGS.xmax, n_x, endpoint=False)
    ys = np.linspace(FLAGS.ymin, FLAGS.ymax, n_y, endpoint=False)

    xv, yv = np.meshgrid(xs, ys)

    xy = np.stack((xv.flatten(), yv.flatten()), axis=1)

    xy = xy + jax.random.uniform(
        k1, minval=0.0, maxval=np.array([[dx, dy]]), shape=(len(xy), 2)
    )

    in_hole = jax.vmap(
        jax.vmap(is_in_hole, in_axes=(0, None)), in_axes=(None, 0), out_axes=1
    )(xy, per_hole_params)

    mask = np.arange(per_hole_params.shape[0], dtype=np.int32).reshape(1, -1)
    mask = mask < n_holes
    in_hole = in_hole * mask
    in_hole = np.any(in_hole, axis=1)

    idxs = jax.random.choice(k2, xy.shape[0], replace=False, p=1 - in_hole, shape=(n,))
    return xy[idxs]


def is_defined(xy, u):
    try:
        u(xy)
        return True
    except Exception as e:
        return False


class SecondOrderTaylorLookup(object):
    def __init__(self, u, x0, d=3):
        x0 = np.array(x0)
        Vg = fa.TensorFunctionSpace(u.function_space().mesh(), "P", 2, shape=(d, 2))
        ug = fa.project(fa.grad(u), Vg, solver_type="mumps")
        ug.set_allow_extrapolation(True)
        Vh = fa.TensorFunctionSpace(u.function_space().mesh(), "P", 2, shape=(d, 2, 2))
        uh = fa.project(fa.grad(ug), Vh, solver_type="mumps")
        uh.set_allow_extrapolation(True)

        u.set_allow_extrapolation(True)
        self.x0s = x0.reshape(len(x0), 2)
        self.u0s = np.array([u(npo.array(xi)) for xi in x0])
        self.g0s = np.array([ug(npo.array(xi)) for xi in x0])
        self.h0s = np.array([uh(npo.array(xi)) for xi in x0])
        u.set_allow_extrapolation(False)

    def __call__(self, x):
        x = x.reshape(-1, 2)
        dists = np.sum((self.x0s.reshape(1, -1, 2) - x.reshape(-1, 1, 2)) ** 2, axis=2)
        _, inds = jax.lax.top_k(-dists, 1)
        x0 = self.x0s[inds]
        u0 = self.u0s[inds]
        g0 = self.g0s[inds]
        h0 = self.h0s[inds]
        return jax.vmap(single_second_order_taylor_eval)(x, x0, u0, g0, h0).squeeze()


@jax.jit
def single_second_order_taylor_eval(xi, x0i, u0, g0, h0, d=3):
    dx = xi - x0i
    return (
        u0.reshape(d)
        + np.matmul(g0.reshape(d, 2), dx.reshape(2, 1)).reshape(d)
        + np.matmul(
            dx.reshape(1, 1, 2), np.matmul(h0.reshape(d, 2, 2), dx.reshape(1, 2, 1))
        ).reshape(d)
        / 2.0
    )


def fenics_to_jax(u, gridsize=300, temp=1.0):
    X, Y = np.meshgrid(
        np.linspace(FLAGS.xmin, FLAGS.xmax, 3 * gridsize),
        np.linspace(FLAGS.ymin, FLAGS.ymax, gridsize),
    )

    XY = list(zip(X.reshape(-1), Y.reshape(-1)))

    mask = [is_defined([x, y], u) for x, y in XY]

    u.set_allow_extrapolation(True)
    U = [u(x, y) for x, y in XY]
    u.set_allow_extrapolation(False)
    U = np.array(U)
    XY = np.array(XY)
    mask = np.array(mask, dtype=np.float32)
    # U = np.array(U).reshape(121, 41, 3)

    def interpolated_function(x, temp=temp):
        # 5-nearest-neighbor interpolation with low-temperature softmax
        x = x.reshape(-1, 2)
        bsize = x.shape[1]
        dists = np.sum((XY - x) ** 2, axis=1)
        _, inds = jax.lax.top_k(-dists, 5)
        dists = dists[inds]
        vals = U[inds]
        is_defined_mask = mask[inds]
        weights = jax.nn.softmax(is_defined_mask * temp / (dists + 1e-14)).reshape(
            -1, 1
        )
        return (weights * vals).sum(axis=0).reshape(3)

    def maybe_vmapped(x):
        if len(x.shape) == 1:
            return interpolated_function(x)
        elif len(x.shape) == 2:
            return jax.vmap(interpolated_function)(x)
        else:
            raise Exception("Wrong shape!")

    return maybe_vmapped


def error_on_coords(fenics_fn, jax_fn, coords=None):
    if coords is None:
        coords = np.array(fenics_fn.function_space().tabulate_dof_coordinates())
    fenics_vals = np.array([fenics_fn(x) for x in coords])
    jax_vals = jax_fn(coords)
    return np.mean((fenics_vals - jax_vals) ** 2)


def plot_solution(u, params):
    _, _, per_hole_params, num_holes = params

    X, Y = npo.meshgrid(
        npo.linspace(FLAGS.xmin, FLAGS.xmax, 300),
        npo.linspace(FLAGS.ymin, FLAGS.ymax, 100),
    )
    Xflat, Yflat = X.reshape(-1), Y.reshape(-1)

    # X, Y = X[valid], Y[valid]
    valid = [is_defined([x, y], u) for x, y in zip(Xflat, Yflat)]

    UV = [
        u(x, y) if is_defined([x, y], u) else npo.array([0.0, 0.0])
        for x, y in zip(Xflat, Yflat)
    ]

    U = npo.array([uv[0] for uv in UV]).reshape(X.shape)
    V = npo.array([uv[1] for uv in UV]).reshape(Y.shape)

    X_, Y_ = npo.meshgrid(
        npo.linspace(FLAGS.xmin, FLAGS.xmax, 60),
        npo.linspace(FLAGS.ymin, FLAGS.ymax, 40),
    )
    Xflat_, Yflat_ = X_.reshape(-1), Y_.reshape(-1)

    # X, Y = X[valid], Y[valid]
    valid_ = [is_defined([x, y], u) for x, y in zip(Xflat_, Yflat_)]
    Xflat_, Yflat_ = Xflat_[valid_], Yflat_[valid_]
    UV_ = [u(x, y) for x, y in zip(Xflat_, Yflat_)]

    U_ = npo.array([uv[0] for uv in UV_])
    V_ = npo.array([uv[1] for uv in UV_])

    speed = npo.linalg.norm(npo.stack([U, V], axis=2), axis=2)

    speed_ = npo.linalg.norm(npo.stack([U_, V_], axis=1), axis=1)

    seed_points = npo.stack(
        [
            FLAGS.xmin * npo.ones(40),
            npo.linspace(FLAGS.ymin + 0.1, FLAGS.ymax - 0.1, 40),
        ],
        axis=1,
    )

    intensity = fa.inner(u, u) + 0.1
    fa.plot(intensity,
            mode="color",
            shading="gouraud",
            edgecolors="k",
            linewidth=0.0,
            cmap="BuPu",
        )

    plt.quiver(Xflat_, Yflat_, U_, V_, speed_ / speed_.max(), alpha=0.7)

    plt.streamplot(
        X,
        Y,
        U,
        V,
        color=speed / speed.max(),
        start_points=seed_points,
        density=10,
        linewidth=0.2,
        arrowsize=0.0,
    )  # , np.sqrt(U**2+V**2))


def main(argv):
    print("non-flag arguments:", argv)

    key, subkey = jax.random.split(jax.random.PRNGKey(FLAGS.seed))

    for i in range(1, 10):
        key, sk1, sk2 = jax.random.split(key, 3)
        params = sample_params(sk1)

        points = sample_points(sk2, 512, params)
        (
            points_on_inlet,
            points_on_outlet,
            points_on_walls,
            points_on_holes,
            points_in_domain,
        ) = points
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.scatter(
            points_on_inlet[:, 0], points_on_inlet[:, 1], label="points on inlet"
        )
        plt.scatter(
            points_on_outlet[:, 0], points_on_outlet[:, 1], label="points on outlet"
        )
        plt.scatter(
            points_on_walls[:, 0], points_on_walls[:, 1], label="points on walls"
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.scatter(
            points_on_holes[:, 0], points_on_holes[:, 1], label="points on holes"
        )
        plt.scatter(
            points_in_domain[:, 0], points_in_domain[:, 1], label="points in domain"
        )
        plt.legend()

        plt.show()


if __name__ == "__main__":
    app.run(main)
