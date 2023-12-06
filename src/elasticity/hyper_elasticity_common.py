import jax
import jax.numpy as np
import numpy as npo
import pdb
from functools import partial
import argparse
from collections import namedtuple
from itertools import product

import matplotlib.pyplot as plt

import fenics as fa

from ..util.jax_tools import tree_unstack

from absl import app
from absl import flags
from ..util import common_flags

FLAGS = flags.FLAGS


def deformation_gradient(x, field_fn):
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
        young_mod = bc_params[0].astype(float)
        poisson_ratio = 0.49
        d = 2
        shear_mod = young_mod / (2 * (1 + poisson_ratio))
        bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))
        F = deformation_gradient(x, field_fn)
        J = np.linalg.det(F)
        Jinv = J ** (-2 / d)
        Ic = np.trace(right_cauchygreen(x, field_fn))
        energy = (shear_mod / 2) * (Jinv * Ic - d) + (bulk_mod / 2) * (J - 1) ** 2

        return energy

    vmap_integrand = jax.vmap(integrand, in_axes=(0, None))
    err = vmap_integrand(points_in_domain, field_fn)

    return err


def loss_top_fn(field_fn, points_on_top, params):
    source_params, bc_params, per_hole_params, n_holes = params
    return (field_fn(points_on_top) - np.array([0., -0.12])) ** 2


def loss_bottom_fn(field_fn, points_on_bottom, params):
    source_params, bc_params, per_hole_params, n_holes = params
    return field_fn(points_on_bottom)** 2


def loss_fn(field_fn, points, params):
    (
        points_on_top,
        points_on_bottom,
        points_on_left,
        points_on_right,
        points_on_holes,
        points_in_domain,
    ) = points
    return (
        {
            "loss_bottom": 1000. * np.mean(loss_bottom_fn(field_fn, points_on_bottom, params)),
            "loss_top": 1000. * np.mean(loss_top_fn(field_fn, points_on_top, params)),
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
        key = jax.random.PRNGKey(FLAGS.seed)

    key, subkey = jax.random.split(key)
    output_tuple = jax.lax.while_loop(sample_params_cond, sample_params_body, (key, True, np.zeros((2,)), np.zeros((2,)), np.zeros((25,5)), 0))

    _, _, source_params, bc_params, per_hole_params, n_holes = output_tuple

    return source_params, bc_params, per_hole_params, n_holes


def sample_params_cond(input_tuple):
    _, feasibility, _, _, _, _ = input_tuple

    return feasibility


def sample_params_body(input_tuple):
    key, _, _, _, _, _ = input_tuple
    key, subkey = jax.random.split(key)

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
        k2, minval=0.9, maxval=1.1, shape=(2,)
    )

    if not FLAGS.max_holes > 0:
        n_holes = 0
        pore_x0y0 = np.zeros((1, 2))
        pore_shapes = np.zeros((1, 2))
        pore_sizes = np.zeros((1, 1))
        per_hole_params = np.concatenate((pore_shapes, pore_x0y0, pore_sizes), axis=1)
        return source_params, bc_params, per_hole_params, n_holes

    # number of pores
    n_holes = FLAGS.max_holes ** 2

    # pore shape
    pore_shape = 0.0 * np.array([
        jax.random.uniform(k3, minval=-0.1, maxval=0.1, shape=(1,)),
        jax.random.uniform(k4, minval=-0.1, maxval=0.1, shape=(1,))]).T

    pore_shapes = np.repeat(pore_shape, n_holes, axis=0)

    # pore location
    spacing = 0.0
    xlow = FLAGS.xmin + spacing * FLAGS.max_hole_size
    xhigh = FLAGS.xmax - spacing * FLAGS.max_hole_size
    ylow = FLAGS.ymin + spacing * FLAGS.max_hole_size
    yhigh = FLAGS.ymax - spacing * FLAGS.max_hole_size

    pore_x0 = np.linspace(xlow, xhigh, FLAGS.max_holes)
    pore_y0 = np.linspace(ylow, yhigh, FLAGS.max_holes)
    pore_x0y0 = np.array(list(product(pore_x0, pore_y0)))

    # pore size
    L0 = pore_x0[1] - pore_x0[0]
    phi = 0.5  # porosity
    r0 = (L0 * np.sqrt(2 * phi) / np.sqrt((2 + pore_shapes[0, 0] ** 2 + pore_shapes[0, 1] ** 2) * np.pi))
    pore_sizes = np.repeat(r0, n_holes, axis=0)[:, None]
    pore_scale = jax.random.uniform(
        k6,
        minval=0.2 * FLAGS.max_hole_size,
        maxval=1.5 * FLAGS.max_hole_size,
        shape=(1,))
    pore_scales = np.repeat(pore_scale, n_holes, axis=0)[:, None]
    pore_sizes = pore_sizes * pore_scales

    # pore feasibility
    t_bar = 0.05
    theta = np.linspace(0, 2 * np.pi, 1_000)
    r_theta = pore_scale * r0 * (1 + pore_shape[0, 0] * np.cos(4 * theta) + pore_shape[0, 1] * np.cos(8 * theta))
    x1 = r_theta * np.cos(theta)
    tmin = (L0 - 2 * np.max(x1)) / L0

    feasibility = (tmin < t_bar)

    per_hole_params = np.concatenate((pore_shapes, pore_x0y0, pore_sizes), axis=1)

    return (subkey, feasibility, source_params, bc_params, per_hole_params, n_holes)


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
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    ratio = (FLAGS.xmax - FLAGS.xmin) / (FLAGS.ymax - FLAGS.ymin)
    n_on_boundary = n
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
def masking_pore_points(key, n, xy, params):
    _, _, per_hole_params, n_holes = params

    in_hole = jax.vmap(
        jax.vmap(is_in_hole, in_axes=(0, None)), in_axes=(None, 0), out_axes=1
    )(xy, per_hole_params)

    mask = np.arange(per_hole_params.shape[0], dtype=np.int32).reshape(1, -1)
    mask = mask < n_holes
    in_hole = in_hole * mask
    in_hole = np.any(in_hole, axis=1)
    # in_hole = np.squeeze(jax.vmap(is_in_hole, in_axes=(0, None))(xy, per_hole_params))

    idxs = jax.random.choice(key, xy.shape[0], replace=False, p=1 - in_hole, shape=(n,))

    return xy[idxs]


@partial(jax.jit, static_argnums=(1,))
def sample_points_top(key, n, params):
    _, _, _, _ = params
    k1, k2 = jax.random.split(key)

    n_tmp = 10 * n
    top_x = jax.random.uniform(
        key, minval=FLAGS.xmin, maxval=FLAGS.xmax, shape=(n_tmp,)
    )
    xy = np.stack([top_x, FLAGS.ymax * np.ones(n_tmp)], axis=1)

    top = masking_pore_points(k2, n, xy, params)
    return top


@partial(jax.jit, static_argnums=(1,))
def sample_points_bottom(key, n, params):
    _, _, per_hole_params, n_holes = params
    k1, k2 = jax.random.split(key)

    n_tmp = 10 * n
    bottom_x = jax.random.uniform(
        key, minval=FLAGS.xmin, maxval=FLAGS.xmax, shape=(n_tmp,)
    )
    xy = np.stack([bottom_x, FLAGS.ymin * np.ones(n_tmp)], axis=1)

    bottom = masking_pore_points(k2, n, xy, params)
    return bottom


@partial(jax.jit, static_argnums=(1,))
def sample_points_left(key, n, params):
    _, _, per_hole_params, n_holes = params
    k1, k2 = jax.random.split(key)

    n_tmp = 10 * n
    left_y = jax.random.uniform(
        key, minval=FLAGS.ymin, maxval=FLAGS.ymax, shape=(n_tmp,)
    )
    xy = np.stack([FLAGS.xmin * np.ones(n_tmp), left_y], axis=1)

    left = masking_pore_points(k2, n, xy, params)
    return left


@partial(jax.jit, static_argnums=(1,))
def sample_points_right(key, n,  params):
    _, _, per_hole_params, n_holes = params
    k1, k2 = jax.random.split(key)

    n_tmp = 10 * n
    right_y = jax.random.uniform(
        key, minval=FLAGS.ymin, maxval=FLAGS.ymax, shape=(n_tmp,)
    )
    xy = np.stack([FLAGS.xmax * np.ones(n_tmp), right_y], axis=1)

    right = masking_pore_points(k2, n, xy, params)
    return right


def is_in_bound(xy):
    return (xy[0] > FLAGS.xmin) * (xy[0] < FLAGS.xmax) * (xy[1] > FLAGS.ymin) * (xy[1] < FLAGS.ymax)


@partial(jax.jit, static_argnums=(1,))
def sample_points_on_pores(key, n, params):
    _, _, per_hole_params, n_holes = params
    n_tmp = int(1.5 * n)

    x = []
    y = []
    for per_hole_param in per_hole_params:

        c = per_hole_param[0: 2]
        xy = per_hole_param[2: 4]
        size = per_hole_param[4]

        thetas = jax.random.uniform(
            key, minval=0.0, maxval=1.0, shape=(n_tmp,)
        ) * 2 * np.pi

        r0 = size * (1.0 + c[0] * np.cos(4 * thetas) + c[1] * np.cos(8 * thetas))
        x.append(xy[0] + r0 * np.cos(thetas))
        y.append(xy[1] + r0 * np.sin(thetas))
    x = np.array(x)
    y = np.array(y)

    xy = np.stack([np.concatenate(x), np.concatenate(y)], axis=1)

    in_bound = jax.vmap(is_in_bound, in_axes=0, out_axes=0)(xy)

    idxs = jax.random.choice(key, xy.shape[0], replace=False, p=in_bound, shape=(n,))

    return xy[idxs]


@partial(jax.jit, static_argnums=(1,))
def sample_points_in_domain(key, n, params):
    _, _, per_hole_params, n_holes = params
    k1, k2, k3 = jax.random.split(key, 3)
    # total number of points is 2 * n
    # so as long as the fraction of volume covered is << 1/2 we are ok
    n_x = 3 * n
    n_y = 3 * n

    xs = jax.random.uniform(k1, minval=FLAGS.xmin, maxval=FLAGS.xmax, shape=(n_x, ))
    ys = jax.random.uniform(k2, minval=FLAGS.ymin, maxval=FLAGS.ymax, shape=(n_y, ))

    xy = np.stack((xs.flatten(), ys.flatten()), axis=1)

    in_hole = jax.vmap(
        jax.vmap(is_in_hole, in_axes=(0, None)), in_axes=(None, 0), out_axes=1
    )(xy, per_hole_params)

    mask = np.arange(per_hole_params.shape[0], dtype=np.int32).reshape(1, -1)
    mask = mask < n_holes
    in_hole = in_hole * mask
    in_hole = np.any(in_hole, axis=1)

    idxs = jax.random.choice(k3, xy.shape[0], replace=False, p=1 - in_hole, shape=(n,))
    return xy[idxs]


def is_defined(xy, u):
    try:
        u(xy)
        return True
    except Exception as e:
        return False


def error_on_coords(fenics_fn, jax_fn, coords=None):
    if coords is None:
        coords = np.array(fenics_fn.function_space().tabulate_dof_coordinates())
    fenics_vals = np.array([fenics_fn(x) for x in coords])
    jax_vals = jax_fn(coords)
    return np.mean((fenics_vals - jax_vals) ** 2)


def plot_solution(u, params=None):
    c = fa.plot(
        u,
        mode="displacement",
    )


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
