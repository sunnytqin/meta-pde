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


FLAGS = flags.FLAGS
flags.DEFINE_float("tmin", 0.0, "PDE initial time")
flags.DEFINE_float("tmax", 0.5, "PDE final time")
flags.DEFINE_float("tmax_nn", -1., "PDE final time passed in for NN")
flags.DEFINE_integer("num_tsteps", 5, "number of time steps for td_burgers")
flags.DEFINE_boolean("sample_time_random", True, "random time sample for NN")
flags.DEFINE_float("max_reynolds", 1e3, "Reynolds number scale")
flags.DEFINE_float("time_scale_deviation", 0.1, "Used to time scale loss")
flags.DEFINE_boolean("td_burger_impose_symmetry", True, "for bc param sampling")
flags.DEFINE_integer("propagatetime_max", 200_000, "maximum iterations before propagate time step")
flags.DEFINE_float("propagatetime_rel", 0.01, "rel val improvment change before propagate time step")
FLAGS.bc_weight = 1.0
FLAGS.max_holes = 0
FLAGS.viz_every = int(5e4)
FLAGS.log_every = int(5e3)
FLAGS.measure_grad_norm_every = int(2e3)
FLAGS.outer_steps = int(1e8)
FLAGS.n_eval = 5

class GroundTruth:
    def __init__(self, fenics_functions_list, timesteps_list):
        self.fenics_functions_list = fenics_functions_list
        self.timesteps_list = np.array(timesteps_list)
        assert type(fenics_functions_list) == list
        self.fenics_function_sample = fenics_functions_list[0]
        self.tsteps = len(self.fenics_functions_list)

    def function_space(self):
        return self.fenics_function_sample.function_space()

    def set_allow_extrapolation(self, boolean):
        for f in self.fenics_functions_list:
            f.set_allow_extrapolation(boolean)

    def __len__(self):
        return self.tsteps

    def __call__(self, x):
        # require time axis aligns with the PDE time stepping
        t_step = np.squeeze(np.argwhere(np.isclose(self.timesteps_list, x[2])))
        fenics_function = self.fenics_functions_list[t_step]
        return fenics_function(x[:-1])

    def __getitem__(self, i):
        return self.fenics_functions_list[i]


def loss_domain_fn(field_fn, points_in_domain, params):
    # ut = Re(uxx + uyy) - (u ux + v uy)
    # vt = Re(vxx + vyy) - (u vx + v vy)
    source_params, bc_params, per_hole_params, n_holes = params

    jac_fn = jax.jacfwd(field_fn)

    def lhs_fn(x):  # need to be time derivative
        return (jac_fn(x)[:, -1]).reshape(2)

    def rhs_fn(x):
        hessian = jax.hessian(field_fn)
        nabla_term = (1./ source_params[0]) * np.trace(hessian(x)[[0, 1], :-1, :-1])
        grad_term = np.matmul(
            jac_fn(x)[:, :-1].reshape(2, 2), field_fn(x).reshape(2, 1)
        ).reshape(2)
        return nabla_term - grad_term

        #uxx = jax.jvp(lambda xi: jax.jvp(field_fn, (xi,), (np.array([1., 0.]),))[1],
        #        (x,), (np.array([1., 0.]),))[1]
        #uyy = jax.jvp(lambda xi: jax.jvp(field_fn, (xi,), (np.array([0., 1.]),))[1],
        #                (x,), (np.array([0., 1.]),))[1]
        #return (1./ source_params[0]) * (uxx + uyy).reshape(2)

    return (jax.vmap(lhs_fn)(points_in_domain) - jax.vmap(rhs_fn)(points_in_domain))**2


def loss_vertical_fn(field_fn, points_on_vertical, params):
    source_params, bc_params, per_hole_params, n_holes = params

    A0 = (np.abs(bc_params[0, 0])).astype(float)
    A1 = (np.abs(bc_params[0, 1])).astype(float)
    sinusoidal_magnitude = A0 * \
                           np.cos(A1 * np.pi * (points_on_vertical[:, 0] - FLAGS.xmin) / (FLAGS.xmax - FLAGS.xmin)) *\
                           np.sin(A1 * np.pi * (points_on_vertical[:, 1] - FLAGS.ymin) / (FLAGS.ymax - FLAGS.ymin))
    zero_magnitude = np.zeros_like(sinusoidal_magnitude)

    return (field_fn(points_on_vertical) - np.stack((zero_magnitude, sinusoidal_magnitude), axis=-1)) ** 2


def loss_horizontal_fn(field_fn, points_on_horizontal, params):
    source_params, bc_params, per_hole_params, n_holes = params

    A0 = (np.abs(bc_params[0, 0])).astype(float)
    A1 = (np.abs(bc_params[0, 1])).astype(float)
    sinusoidal_magnitude = A0 * \
                           np.sin(A1 * np.pi * (points_on_horizontal[:, 0] - FLAGS.xmin) / (FLAGS.xmax - FLAGS.xmin)) * \
                           np.cos(A1 * np.pi * (points_on_horizontal[:, 1] - FLAGS.ymin) / (FLAGS.ymax - FLAGS.ymin))
    zero_magnitude = np.zeros_like(sinusoidal_magnitude)

    return (field_fn(points_on_horizontal) - np.stack((sinusoidal_magnitude, zero_magnitude), axis=-1)) ** 2


def loss_initial_fn(field_fn, points_initial, params):
    source_params, bc_params, per_hole_params, n_holes = params

    A0 = (np.abs(bc_params[0, 0])).astype(float)
    A1 = (np.abs(bc_params[0, 1])).astype(float)
    sinusoidal_magnitude_x = A0 * \
                             np.sin(A1 * np.pi * (points_initial[:, 0] - FLAGS.xmin) / (FLAGS.xmax - FLAGS.xmin)) * \
                             np.cos(A1 * np.pi * (points_initial[:, 1] - FLAGS.ymin) / (FLAGS.ymax - FLAGS.ymin))
    sinusoidal_magnitude_y = A0 * \
                             np.cos(A1 * np.pi * (points_initial[:, 0] - FLAGS.xmin) / (FLAGS.xmax - FLAGS.xmin)) * \
                             np.sin(A1 * np.pi * (points_initial[:, 1] - FLAGS.ymin) / (FLAGS.ymax - FLAGS.ymin))
    return (field_fn(points_initial) -
            np.stack((sinusoidal_magnitude_x, sinusoidal_magnitude_y), axis=-1)) ** 2


def loss_time_scale(field_fn, points_in_domain, params):

    def taylor_val(x):
        deviation = np.array([0.0, 0.0, FLAGS.time_scale_deviation])
        # taylor expansion val
        def filed_fn_x(points):
            return field_fn(points).reshape(2)[0]

        # compute taylor expansion for x component
        jacobian = jax.jacfwd(filed_fn_x)
        hessian = jax.hessian(filed_fn_x)
        x_new = (
            (filed_fn_x(x) + np.dot(jacobian(x), deviation) + 0.5 * np.dot(np.transpose(deviation), np.dot(hessian(x), deviation))).astype(float)
        )

        def filed_fn_y(points):
            return field_fn(points).reshape(2)[1]

        # compute taylor expansion for y component
        jacobian = jax.jacfwd(filed_fn_y)
        hessian = jax.hessian(filed_fn_y)
        y_new = (
            (filed_fn_y(x) + np.dot(jacobian(x), deviation) + 0.5 * np.dot(np.transpose(deviation), np.dot(hessian(x), deviation))).astype(float)
        )
        return np.array([x_new, y_new])

    def model_val(x):
        x1 = x + np.array([0.0, 0.0, FLAGS.time_scale_deviation])
        return field_fn(x1)

    # compare model value with taylor expanded val
    return (jax.vmap(model_val)(points_in_domain) - jax.vmap(taylor_val)(points_in_domain)) ** 2

def loss_fn(field_fn, points, params):
    (
        points_on_vertical,
        points_on_horizontal,
        points_initial,
        points_on_holes,
        points_in_domain,
    ) = points
    #points_noslip = np.concatenate([points_on_walls, points_on_holes])

    return (
        {
            "loss_initial": np.mean(loss_initial_fn(field_fn, points_initial, params)),
            "loss_vertical": np.mean(loss_vertical_fn(field_fn, points_on_vertical, params)),
            "loss_horizontal": np.mean(loss_horizontal_fn(field_fn, points_on_horizontal, params)),
            #"loss_time_scale": np.mean(loss_time_scale(field_fn, points_in_domain, params)),
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

    if FLAGS.td_burger_impose_symmetry:
        bc_params = FLAGS.bc_scale * jax.random.uniform(
            k2, minval=0.0, maxval=1.5, shape=(1, 1,)
        )
        bc_params = np.concatenate([bc_params, np.array([[1.]])], axis=1)
    else:
        bc_params = FLAGS.bc_scale * jax.random.uniform(
            k2, minval=0.0, maxval=1.5, shape=(1, 2,)
        )

    if not FLAGS.max_holes > 0:
        n_holes = 0
        pore_x0y0 = np.zeros((1, 2))
        pore_shapes = np.zeros((1, 2))
        pore_sizes = np.zeros((1, 1))
        per_hole_params = np.concatenate((pore_shapes, pore_x0y0, pore_sizes), axis=1)
        return source_params, bc_params, per_hole_params, n_holes

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
    n_on_walls = n
    n_on_holes = n // 2 - n_on_walls #- n_on_inlet - n_on_outlet
    points_on_vertical = sample_points_on_vertical(k1, n_on_walls, params)
    points_on_horizontal = sample_points_on_horizontal(k2, n_on_walls, params)
    points_initial = sample_points_initial(k3, n_on_walls, params)
    points_in_domain = sample_points_in_domain(k5, n, params)
    if FLAGS.max_holes > 0:
        points_on_holes = sample_points_on_pores(k4, n_on_holes, params)
    else:
        points_on_holes = np.array([points_in_domain[0]])


    return (
        points_on_vertical,
        points_on_horizontal,
        points_initial,
        points_on_holes,
        points_in_domain,
    )


@partial(jax.jit, static_argnums=(1,))
def sample_points_on_vertical(key, n, params):
    k1, k2 = jax.random.split(key)
    n_scaled = n // (FLAGS.num_tsteps - 1)
    n_scaled = n_scaled // 2
    _, _, per_hole_params, n_holes = params
    y = (np.linspace(FLAGS.ymin, FLAGS.ymax, n_scaled, endpoint=False) + jax.random.uniform(
        k1, minval=0.0, maxval=(FLAGS.ymax - FLAGS.ymin) / n_scaled, shape=(1,)
    )).reshape(n_scaled, 1)
    t = sample_time(k2, n_scaled)
    y = np.tile(y, (FLAGS.num_tsteps - 1, 1))
    lhs_x = (FLAGS.xmin * np.ones(len(t))).reshape(-1, 1)
    rhs_x = (FLAGS.xmax * np.ones(len(t))).reshape(-1, 1)
    lhs_xy_t = np.concatenate([lhs_x, y, t], axis=1)
    rhs_xy_t = np.concatenate([rhs_x, y, t], axis=1)

    return np.concatenate((lhs_xy_t, rhs_xy_t))


def sample_points_on_horizontal(key, n, params):
    return sample_points_on_vertical(key, n, params)[:, [1, 0, 2]]


# not used and thus no changed has been made
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


# new
@partial(jax.jit, static_argnums=(1,))
def sample_points_on_pores(key, n, params):
    _, _, per_hole_params, n_holes = params
    k1, k2, k3 = jax.random.split(key, 3)

    mask = np.arange(per_hole_params.shape[0], dtype=np.int32)
    mask = mask < n_holes

    total_sizes = np.sum(per_hole_params[:, -1] ** 2 * mask)

    dpore = np.linspace(0.0, total_sizes, n, endpoint=False)
    dpore = dpore + jax.random.uniform(
        k1, minval=0.0, maxval=(total_sizes / n), shape=(1,)
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
        k2, n * FLAGS.max_holes * 2, replace=False, p=p, shape=(n,)
    )

    tmp = np.tile(hole_points, (FLAGS.max_holes * 2, 1))[idxs]

    t = sample_time(k3, n)

    tmp = np.tile(tmp, (len(t) // len(tmp), 1))
    xy_t = np.concatenate([tmp, t], axis=1)

    return xy_t


# new
@partial(jax.jit, static_argnums=(1,))
def sample_points_in_domain(key, n, params):
    _, _, per_hole_params, n_holes = params
    k1, k2, k3 = jax.random.split(key, 3)
    ratio = (FLAGS.xmax - FLAGS.xmin) / (FLAGS.ymax - FLAGS.ymin)
    # rescale n according to time steps
    n_scaled = n // (FLAGS.num_tsteps - 1)
    # total number of points is 2 * n
    # so as long as the fraction of volume covered is << 1/2 we are ok
    n_x = npo.int32(npo.sqrt(2) * npo.sqrt(n_scaled) * npo.sqrt(ratio))
    n_y = npo.int32(npo.sqrt(2) * npo.sqrt(n_scaled) / npo.sqrt(ratio))
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

    idxs = jax.random.choice(k2, xy.shape[0], replace=False, p=1 - in_hole, shape=(n_scaled,))
    xy = np.array(xy[idxs])

    t = sample_time(k3, n_scaled)

    xy = np.tile(xy, (len(t) // len(xy), 1))
    xy_t = np.concatenate([xy, t], axis=1)

    assert len(xy_t) == n_scaled * (FLAGS.num_tsteps - 1)

    return xy_t


# new
def sample_points_initial(key, n, params):
    points_in_domain = sample_points_in_domain(key, n * (FLAGS.num_tsteps - 1), params)[0: n, :]
    t = np.zeros((points_in_domain.shape[0], 1))
    return np.concatenate([points_in_domain[:, :-1], t], axis=1)


def sample_time(key, n):
    tmax = jax.lax.cond(FLAGS.tmax_nn > 0,
                        lambda _: FLAGS.tmax_nn,
                        lambda _: FLAGS.tmax,
                        operand=None)
    if FLAGS.sample_time_random:
        t = np.linspace(FLAGS.tmin, tmax, FLAGS.num_tsteps, endpoint=False)\
            - jax.random.uniform(
            key, minval=-(tmax - FLAGS.tmin) / n, maxval=0.0, shape=(1, )
            )

        t = np.repeat(t[:-1], n).reshape((FLAGS.num_tsteps - 1) * n, 1)  # excluding last points
    else:
        t = np.linspace(FLAGS.tmin, tmax, FLAGS.num_tsteps, endpoint=False)
        t = np.repeat(t[1:], n).reshape((FLAGS.num_tsteps - 1) * n, 1)  # excluding initial points
    return t


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


def plot_solution(u_list, params, t_val=None):
    if type(u_list) == GroundTruth:
        if t_val is None: # always plot the midpoint is no time is specified
            n_plots = min(len(u_list), 2)
            plot_idx = npo.unique(npo.linspace(0, len(u_list), 2, endpoint=False, dtype=int)[1:])
            print(f'plotting Ground Truth at t = {u_list.timesteps_list[plot_idx]}')
            plot_solution_snapshot(u_list[np.squeeze(plot_idx)])

    else:
        plot_solution_snapshot(u_list)


def plot_solution_snapshot(u):
    intensity = fa.inner(u, u)
    fa.plot(intensity,
            mode="color",
            shading="gouraud",
            edgecolors="k",
            linewidth=0.0,
            cmap="BuPu",
            )
    fa.plot(u)


def main(argv):
    print("non-flag arguments:", argv)

    key, subkey = jax.random.split(jax.random.PRNGKey(FLAGS.seed))

    for i in range(1, 8):
        key, sk1, sk2 = jax.random.split(key, 3)
        params = sample_params(sk1)

        points = sample_points(sk2, 512, params)
        (
            points_on_vertical,
            points_on_horizontal,
            points_initial,
            points_on_holes,
            points_in_domain,
        ) = points

        plt.figure()
        plt.subplot(1, 1, 1)
        plt.scatter(
            points_on_vertical[:, 0], points_on_vertical[:, 1], label="points on vertical walls"
        )
        plt.scatter(
            points_on_horizontal[:, 0], points_on_horizontal[:, 1], label="points on horizontal walls"
        )
        plt.scatter(
            points_initial[:, 0], points_initial[:, 1], label="points initial"
        )
        plt.scatter(
            points_on_holes[:, 0], points_on_holes[:, 1], label="points on holes"
        )
        plt.scatter(
            points_in_domain[:, 0], points_in_domain[:, 1], label="points in domain"
        )
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

        t_bins = np.linspace(FLAGS.tmin, FLAGS.tmax, 2 * FLAGS.num_tsteps, endpoint=True)

        plt.figure(figsize=(15, 8))
        plt.subplot(5, 1, 1)
        plt.hist(points_on_vertical[:, 2], bins=t_bins, label="points on vertical walls")
        plt.legend()

        plt.subplot(5, 1, 2)
        plt.hist(points_on_horizontal[:, 2], bins=t_bins,label="points on horizontal walls")
        plt.legend()

        plt.subplot(5, 1, 3)
        plt.hist(points_initial[:, 2], bins=t_bins, label="points initial: " + str(set(points_initial[:, 2])))
        plt.legend()

        plt.subplot(5, 1, 4)
        plt.hist(points_on_holes[:, 2], bins=t_bins, label="points on holes")
        plt.legend()

        plt.subplot(5, 1, 5)
        plt.hist(points_in_domain[:, 2], bins=t_bins, label="points in domain")
        plt.legend()

        plt.show()

if __name__ == "__main__":
    flags.DEFINE_float("xmin", -1.0, "scale on random uniform bc")
    flags.DEFINE_float("xmax", 1.0, "scale on random uniform bc")
    flags.DEFINE_float("ymin", -1.0, "scale on random uniform bc")
    flags.DEFINE_float("ymax", 1.0, "scale on random uniform bc")
    flags.DEFINE_integer("max_holes", 12, "scale on random uniform bc")
    flags.DEFINE_float("max_hole_size", 0.4, "scale on random uniform bc")

    app.run(main)
