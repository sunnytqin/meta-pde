import jax
import jax.numpy as np
import numpy as npo
import pdb
from functools import partial
import argparse
from collections import namedtuple

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

FLAGS = flags.FLAGS

if __name__ == "__main__":
    flags.DEFINE_float("xmin", 0.0, "scale on random uniform bc")
    flags.DEFINE_float("xmax", 1.0, "scale on random uniform bc")
    flags.DEFINE_float("ymin", 0.0, "scale on random uniform bc")
    flags.DEFINE_float("ymax", 1.0, "scale on random uniform bc")
    flags.DEFINE_integer("max_holes", 1, "scale on random uniform bc")
    flags.DEFINE_float("max_hole_size", 2.0, "scale on random uniform bc")
    flags.DEFINE_boolean("stokes_nonlinear", False, "if True, make nonlinear")


def deformation_gradient(x, field_fn):
    assert len(x.shape) == 1
    jac = jax.jacfwd(lambda x: field_fn(x).squeeze())(x)
    F = (np.identity(2) + jac)
    return F


def right_cauchygreen(x, field_fn):
    F = deformation_gradient(x, field_fn)
    return np.matmul(F, F.transpose())


def loss_domain_fn(field_fn, points_in_domain, params):
    # force div(grad(u) - p * I) = 0
    source_params, bc_params, per_hole_params, n_holes = params

    def integrand(x, field_fn):
        # energy density
        young_mod = 1.
        poisson_ratio = 0.49
        d = 2
        shear_mod = young_mod / (2 * (1 + poisson_ratio))
        bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))
        F = deformation_gradient(x, field_fn)
        J = np.linalg.det(F)
        Jinv = J **(-2/d)
        Ic = np.trace(right_cauchygreen(x, field_fn))
        energy = (shear_mod / 2) * (Jinv * Ic - d) + (bulk_mod / 2) * (J - 1) ** 2

        # body force
        u = field_fn(x)
        b = np.array([0.0, -0.5])
        body_force = np.dot(u, b)
        return energy - body_force

    vmap_integrand = jax.vmap(integrand, in_axes=(0, None))
    err = vmap_integrand(points_in_domain, field_fn)

    return err


@partial(jax.jit, static_argnums=(0,))
def loss_top_fn(field_fn, points_on_top, params):
    source_params, bc_params, per_hole_params, n_holes = params
    g_x = bc_params[0]
    g_y = bc_params[1]
    sigma = jax.vmap(lambda x: sigma_fn(x, field_fn))(points_on_top)
    normal = np.array([0., 1.])
    err = jax.vmap(lambda x: np.matmul(x, normal) - np.array([g_x, g_y]))(sigma)
    err = err**2
    return err


def loss_bottom_fn(field_fn, points_on_bottom, params):
    source_params, bc_params, per_hole_params, n_holes = params
    return field_fn(points_on_bottom)** 2


def loss_left_fn(field_fn, points_on_left, params):
    source_params, bc_params, per_hole_params, n_holes = params
    sigma = jax.vmap(lambda x: sigma_fn(x, field_fn))(points_on_left)
    normal = np.array([-1., 0.])
    err = jax.vmap(lambda x: np.matmul(x, normal) - np.array([0, 0]))(sigma)
    err = err ** 2
    return err


def loss_right_fn(field_fn, points_on_right, params):
    source_params, bc_params, per_hole_params, n_holes = params
    sigma = jax.vmap(lambda x: sigma_fn(x, field_fn))(points_on_right)
    normal = np.array([1., 0.])
    err = jax.vmap(lambda x: np.matmul(x, normal) - np.array([0, 0]))(sigma)
    err = err ** 2
    return err


def loss_in_hole_fn(field_fn, points_on_holes, params):
    source_params, bc_params, per_hole_params, n_holes = params
    c, xy, size = per_hole_params

    def parametrized_x(theta):
        r0 = size * (1.0 + c[0] * np.cos(4 * theta) + c[1] * np.cos(8 * theta))
        x = xy[0] + r0 * np.cos(theta)
        return x.squeeze()

    def parametrized_y(theta):
        r0 = size * (1.0 + c[0] * np.cos(4 * theta) + c[1] * np.cos(8 * theta))
        y = xy[1] + r0 * np.sin(theta)
        return y.squeeze()

    def find_normal_vec(theta):
        norm_x_fn = jax.grad(parametrized_x)
        norm_y_fn = jax.grad(parametrized_y)

        dx_dtheta = norm_x_fn(theta)
        dy_dtheta = norm_y_fn(theta)

        slope = dy_dtheta / dx_dtheta
        norm_slope = -1. / slope

        u = 1
        v = norm_slope
        n = np.linalg.norm([u, v])
        u /= n
        v /= n

        return u, v

    vmap_find_normal_vec = jax.vmap(find_normal_vec)

    cos_thetas = points_on_holes[:, 0] - xy[0]
    sin_thetas = points_on_holes[:, 1] - xy[1]
    tan_thetas = sin_thetas / cos_thetas
    thetas = np.arctan(tan_thetas)

    thetas = (jax.vmap(lambda cos_theta, theta:
                     jax.lax.cond(
                         cos_theta > 0, lambda x: x, lambda x: x - np.pi * np.sign(x), theta
                     )
                     )(cos_thetas, thetas)).flatten()

    u, v = vmap_find_normal_vec(thetas)
    normal = np.stack((u, v)).transpose()

    sigma = jax.vmap(lambda x: sigma_fn(x, field_fn))(points_on_holes)

    err = jax.vmap(lambda x, y: np.matmul(x, y) - np.array([0, 0]), in_axes=(0, 0))(sigma, normal)
    err = err ** 2
    return err


def loss_fn(field_fn, points, params):
    (
        points_on_top,
        points_on_bottom,
        points_on_left,
        points_on_right,
        points_on_holes,
        points_in_domain,
    ) = points
    points_in_domain = np.concatenate([points_in_domain,
                                       points_on_holes,
                                       points_on_top,
                                       points_on_left,
                                       points_on_right
                                       ])
    return (
        {
            "loss_bottom": 5000. * np.mean(loss_bottom_fn(field_fn, points_on_bottom, params)),
        },
        {
            "loss_domain": np.sum(loss_domain_fn(field_fn, points_in_domain, params)),
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

    source_params = jax.random.uniform(k1, shape=(2,), minval=1 / 4, maxval=3.0 / 4)

    bc_params = FLAGS.bc_scale * jax.random.uniform(
        k2, minval=-5.0, maxval=5.0, shape=(2,)
    )

    if not FLAGS.max_holes > 0:
        n_holes = 0
        pore_x0y0 = np.zeros((1, 2))
        pore_shapes = np.zeros((1, 2))
        pore_sizes = np.zeros((1, 1))
        per_hole_params = np.concatenate((pore_shapes, pore_x0y0, pore_sizes), axis=1)
        return source_params, bc_params, per_hole_params, n_holes

    n_holes = 1

    pore_shapes = jax.random.uniform(
        k4, minval=-0.1, maxval=0.1, shape=(2,)
    )

    pore_sizes = jax.random.uniform(
        k6,
        minval=0.8 * FLAGS.max_hole_size,
        maxval=FLAGS.max_hole_size,
        shape=(1,),
    )

    pore_x0y0 = np.array([(FLAGS.xmax + FLAGS.xmin)/2, (FLAGS.ymax + FLAGS.ymin)/2])

    per_hole_params = (pore_shapes, pore_x0y0, pore_sizes)

    return source_params, bc_params, per_hole_params, n_holes


def is_in_hole(xy, pore_params, tol=1e-7):
    c, xy0, size = pore_params
    vector = xy - xy0
    theta = np.arctan2(*vector)
    length = np.linalg.norm(vector)
    r0 = size * (1.0 + c[0] * np.cos(4 * theta) + c[1] * np.cos(8 * theta))
    return r0 > length + tol


@partial(jax.jit, static_argnums=(1,))
def sample_points(key, n, params):
    _, _, per_hole_params, n_holes = params
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    ratio = (FLAGS.xmax - FLAGS.xmin) / (FLAGS.ymax - FLAGS.ymin)
    n_on_boundary = n // 2
    points_on_top = sample_points_top(k1, n_on_boundary, params)
    points_on_bottom = sample_points_bottom(k2, n_on_boundary, params)
    points_on_left = sample_points_left(k3, n_on_boundary, params)
    points_on_right = sample_points_right(k4, n_on_boundary, params)
    if FLAGS.max_holes > 0:
        points_on_holes = sample_points_on_pores(k5, n_on_boundary, params)
    else:
        points_on_holes = points_on_top
    points_in_domain = sample_points_in_domain(k6, n, params)
    return (
        points_on_top,
        points_on_bottom,
        points_on_left,
        points_on_right,
        points_on_holes,
        points_in_domain,
    )


@partial(jax.jit, static_argnums=(1,))
def sample_points_top(key, n, params):
    _, _, _, _ = params
    top_x = np.linspace(FLAGS.xmin, FLAGS.xmax, n, endpoint=False) + jax.random.uniform(
        key, minval=0.0, maxval=(FLAGS.xmax - FLAGS.xmin) / n, shape=(1,)
    )
    top = np.stack([top_x, FLAGS.ymax * np.ones(n)], axis=1)
    return top


@partial(jax.jit, static_argnums=(1,))
def sample_points_bottom(key, n, params):
    _, _, _, _ = params
    bottom_x = np.linspace(FLAGS.xmin, FLAGS.xmax, n, endpoint=False) + jax.random.uniform(
        key, minval=0.0, maxval=(FLAGS.xmax - FLAGS.xmin) / n, shape=(1,)
    )
    bottom = np.stack([bottom_x, FLAGS.ymin * np.ones(n)], axis=1)
    return bottom


@partial(jax.jit, static_argnums=(1,))
def sample_points_left(key, n, params):
    _, _, per_hole_params, n_holes = params
    k1, k2 = jax.random.split(key)
    left_y = np.linspace(
        FLAGS.ymin, FLAGS.ymax, n // 2, endpoint=False
    ) + jax.random.uniform(
        k1, minval=0.0, maxval=(FLAGS.ymax - FLAGS.ymin) / (n // 2), shape=(1,)
    )
    left = np.stack([FLAGS.xmin * np.ones(n // 2), left_y], axis=1)
    return left


@partial(jax.jit, static_argnums=(1,))
def sample_points_right(key, n,  params):
    _, _, per_hole_params, n_holes = params
    k1, k2 = jax.random.split(key)
    right_y = np.linspace(
        FLAGS.ymin, FLAGS.ymax, n - n // 2, endpoint=False
    ) + jax.random.uniform(
        k2, minval=0.0, maxval=(FLAGS.ymax - FLAGS.ymin) / (n - n // 2), shape=(1,)
    )
    right = np.stack([FLAGS.xmax * np.ones(n - n // 2), right_y], axis=1)
    return right


@partial(jax.jit, static_argnums=(1,))
def sample_points_on_pores(key, n, params):
    _, _, per_hole_params, n_holes = params

    c, xy, size = per_hole_params

    thetas = jax.random.uniform(
        key, minval=0.0, maxval=1.0, shape=(n,)
    ) * 2 * np.pi

    r0 = size * (1.0 + c[0] * np.cos(4 * thetas) + c[1] * np.cos(8 * thetas))
    x = xy[0] + r0 * np.cos(thetas)
    y = xy[1] + r0 * np.sin(thetas)

    return np.concatenate([x[:, None], y[:, None]], axis=1)


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

    in_hole = np.squeeze(jax.vmap(is_in_hole, in_axes=(0, None))(xy, per_hole_params))
    #print("in_hole shape: ", in_hole.shape, xy.shape[0])

    #in_hole = np.any(in_hole, axis=1)

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


def error_on_coords(fenics_fn, jax_fn, coords=None):
    if coords is None:
        coords = np.array(fenics_fn.function_space().tabulate_dof_coordinates())
    fenics_vals = np.array([fenics_fn(x) for x in coords])
    jax_vals = jax_fn(coords)
    return np.mean((fenics_vals - jax_vals) ** 2)


def plot_solution(u, params):
    _, _, per_hole_params, num_holes = params
    c = fa.plot(
        u,
        mode="displacement",
    )
    #cb = plt.colorbar(c, shrink=.8)
    #cb.set_label('Displacement', size=6, c='b')
    #cb.ax.tick_params(labelsize=6, color='blue')


def main(argv):
    print("non-flag arguments:", argv)

    key, subkey = jax.random.split(jax.random.PRNGKey(FLAGS.seed))

    for i in range(1, 5):
        key, sk1, sk2 = jax.random.split(key, 3)
        params = sample_params(sk1)

        points = sample_points(sk2, 512, params)
        (
            points_on_top,
            points_on_bottom,
            points_on_left,
            points_on_right,
            points_on_holes,
            points_in_domain,
        ) = points
        plt.figure()
        plt.subplot(1, 1, 1)
        plt.scatter(
            points_on_top[:, 0], points_on_top[:, 1], label="points on top"
        )
        plt.scatter(
            points_on_bottom[:, 0], points_on_bottom[:, 1], label="points on bottom"
        )
        plt.scatter(
            points_on_left[:, 0], points_on_left[:, 1], label="points on left"
        )
        plt.scatter(
            points_on_right[:, 0], points_on_right[:, 1], label="points on right"
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
