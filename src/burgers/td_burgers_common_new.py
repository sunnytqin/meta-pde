import jax
import jax.numpy as np
import numpy as npo
import pdb
from functools import partial
import argparse
import matplotlib.pyplot as plt

import fenics as fa
import importlib

from absl import app
from absl import flags
from ..util import common_flags


FLAGS = flags.FLAGS
flags.DEFINE_float("tmin", 0.0, "PDE initial time")
flags.DEFINE_float("tmax", 1.0, "PDE final time")
flags.DEFINE_integer("num_tsteps", 101, "number of time steps for td_burgers")
flags.DEFINE_integer("sample_tsteps", 64, "number of time steps for td_burgers")
flags.DEFINE_boolean("sample_time_random", True, "random time sample for NN")
flags.DEFINE_float("max_reynolds", 100, "Reynolds number scale")
flags.DEFINE_string("burgers_pde", "default", "types of burgers equation")


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
        t_step = np.argwhere(np.isclose(self.timesteps_list, x[-1]))[0, 0]
        fenics_function = self.fenics_functions_list[t_step]
        return fenics_function(x[:-1])

    def __getitem__(self, i):
        return self.fenics_functions_list[i]


def loss_domain_fn(field_fn, points_in_domain, params):
    # ut = Re(uxx + uyy) - (u ux + v uy)
    # vt = Re(vxx + vyy) - (u vx + v vy)

    # ut = Re(uxx) - u ux
    source_params, ic_params = params

    #def lhs_fn(x):  # need to be time derivative
    #    return jac_fn(x)[1]
    def rhs_fn(x):
        jac_fn = jax.jacfwd(field_fn)
        jac_fn_x = lambda x: jac_fn(x)[0]
        hessian_fn = jax.jacfwd(jac_fn_x)
        jac_fn_val = jac_fn(x)
        time_term = jac_fn_val[1]

        #hessian = jax.hessian(field_fn)
        hessian = hessian_fn(x)[0]
        nabla_term = (1./ source_params[0]) * hessian
        grad_term = jac_fn_val[0] * field_fn(x)
        return time_term - (nabla_term - grad_term)

    return (jax.vmap(rhs_fn)(points_in_domain)) ** 2
    #return (jax.vmap(lhs_fn)(points_in_domain) - jax.vmap(rhs_fn)(points_in_domain))**2


def loss_fn(field_fn, points, params):
    (
        points_on_left,
        points_on_right,
        points_initial,
        points_in_domain,
    ) = points
    loss_fn_lib = importlib.import_module(f'.burgers_formulation.{FLAGS.burgers_pde}', package='src.burgers')
    loss_initial_fn = loss_fn_lib.loss_initial_fn
    loss_left_fn = loss_fn_lib.loss_left_fn
    loss_right_fn = loss_fn_lib.loss_right_fn

    return (
        {
            "loss_initial": np.mean(loss_initial_fn(field_fn, points_initial, params)),
            "loss_left": np.mean(loss_left_fn(field_fn, points_on_left, params)),
            "loss_right": np.mean(loss_right_fn(field_fn, points_on_right, params)),
        },
        {
            "loss_domain": np.mean(loss_domain_fn(field_fn, points_in_domain, params)),
        },
    )


@jax.jit
def sample_params(key):
    if FLAGS.fixed_num_pdes is not None:
        #key = jax.random.PRNGKey(
        #    jax.random.randint(
        #        key, (1,), np.array([0]), np.array([FLAGS.fixed_num_pdes])
        #    )[0]
        #)
        key = jax.random.PRNGKey(FLAGS.seed)

    k1, k2, k3 = jax.random.split(key, 3)

    # These keys will all be 0 if we're not varying that factor
    k1 = k1 * FLAGS.vary_source
    k2 = k2 * FLAGS.vary_ic

    source_params = FLAGS.max_reynolds * jax.random.uniform(k1, shape=(1,), minval=0.8, maxval=1.)
    ic_params = jax.random.uniform(k2, shape=(2,), minval=-2.0, maxval=2.0)

    return source_params, ic_params


@partial(jax.jit, static_argnums=(1,))
def sample_points(key, n, params):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    points_on_left = sample_points_on_left(k2, n, params)
    points_on_right = sample_points_on_right(k2, n, params)
    points_initial = sample_points_initial(k3, n, params)
    points_in_domain = sample_points_in_domain(k5, n, params)

    return (
        points_on_left,
        points_on_right,
        points_initial,
        points_in_domain,
    )


@partial(jax.jit, static_argnums=(1,))
def sample_points_on_left(key, n, params):
    k1, k2 = jax.random.split(key)
    n_scaled = 1
    x = FLAGS.xmin * np.ones((n_scaled, 1))
    t = sample_time(k2, n_scaled)
    x = np.tile(x, (FLAGS.sample_tsteps - 1, 1))
    x_t = np.concatenate([x, t], axis=1)

    return x_t


@partial(jax.jit, static_argnums=(1,))
def sample_points_on_right(key, n, params):
    k1, k2 = jax.random.split(key)
    n_scaled = 1
    x = FLAGS.xmax * np.ones((n_scaled, 1))
    t = sample_time(k2, n_scaled)
    x = np.tile(x, (FLAGS.sample_tsteps - 1, 1))
    x_t = np.concatenate([x, t], axis=1)

    return x_t


@partial(jax.jit, static_argnums=(1, ))
def sample_points_in_domain(key, n, params):
    k1, k2 = jax.random.split(key, 2)
    # rescale n according to time steps
    n_scaled = n // (FLAGS.sample_tsteps - 1)
    n_sample = n_scaled * (FLAGS.sample_tsteps - 1)

    xs = jax.random.uniform(k1, minval=FLAGS.xmin, maxval=FLAGS.xmax, shape=(n_sample, ))
    t = sample_time(k2, n_scaled)
    xs_t = np.concatenate([xs[:, None], t], axis=1)

    return xs_t


def sample_points_initial(key, n, params):
    points_in_domain = sample_points_in_domain(key, n, params)
    points_boundary = np.array([FLAGS.xmin, FLAGS.xmax])[:, None]
    points_in_domain = np.concatenate([points_in_domain[:, 0: 1], points_boundary], axis=0)
    t = np.zeros((points_in_domain.shape[0], 1))
    return np.concatenate([points_in_domain, t], axis=1)


def sample_time(key, n):
    if FLAGS.sample_time_random:
        num_tsteps = FLAGS.sample_tsteps - 1
        t = jax.random.uniform(key, (num_tsteps * n, 1), minval=FLAGS.tmin, maxval=FLAGS.tmax)
    else:
        num_tsteps = FLAGS.sample_tsteps - 1
        t = np.linspace(FLAGS.tmin, FLAGS.tmax, num_tsteps, endpoint=False)
        t = np.repeat(t[1:], n).reshape(num_tsteps * n, 1)  # excluding initial points
    return t


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


def plot_solution(u_list, params=None):
    if type(u_list) == GroundTruth:
        fenics_vals = []
        for t in range(FLAGS.num_tsteps):
            u_list.set_allow_extrapolation(True)
            x_coords = np.linspace(FLAGS.xmin, FLAGS.xmax, FLAGS.validation_points)

            fenics_vals.append(
                np.array([u_list[t](xi) for xi in x_coords])
            )
        fenics_vals = np.array(fenics_vals)
        clr = plt.imshow(fenics_vals.T, cmap='rainbow', interpolation='none', aspect='auto')
        plt.xticks(np.linspace(0, FLAGS.num_tsteps, 11), np.linspace(0, FLAGS.tmax, 11), fontsize=4)
        plt.yticks(np.linspace(0, FLAGS.validation_points, 11), np.linspace(FLAGS.xmin, FLAGS.xmax, 11), fontsize=4)
        plt.ylabel('position', fontsize=4)
        cbar = plt.colorbar(clr)
        cbar.ax.tick_params(labelsize=4)

    else:
        raise TypeError


def plot_solution_snapshot(u):
    d = fa.plot(u,)
    plt.grid(True)


def main(argv):
    print("non-flag arguments:", argv)

    key, subkey = jax.random.split(jax.random.PRNGKey(FLAGS.seed))

    for i in range(1, 8):
        key, sk1, sk2 = jax.random.split(key, 3)
        params = sample_params(sk1)

        points = sample_points(sk2, 512, params)
        (
            points_on_left,
            points_on_right,
            points_initial,
            points_in_domain,
        ) = points

        plt.figure()
        plt.subplot(1, 1, 1)
        plt.scatter(
            points_on_left[:, 0], points_on_left[:, 1], label="points on left walls"
        )
        plt.scatter(
            points_on_right[:, 0], points_on_right[:, 1], label="points on right walls"
        )
        plt.scatter(
            points_initial[:, 0], points_initial[:, 1], label="points initial"
        )
        plt.scatter(
            points_in_domain[:, 0], points_in_domain[:, 1], label="points in domain"
        )
        plt.xlabel("X")
        plt.ylabel("t")
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


