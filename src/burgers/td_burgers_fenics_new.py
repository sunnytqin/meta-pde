"""
    Solve the burger's equation using fenics

    PDE: https://people.math.sc.edu/Burkardt/fenics_src/burgers_time_viscous/burgers_time_viscous.py
"""

import fenics as fa
import matplotlib.pyplot as plt
import mshr
import numpy as np
import pdb
import argparse
import jax
from collections import namedtuple
import os
import imageio
import importlib

from absl import app
from absl import flags
from ..util import common_flags
from ..util.trainer_util import read_fenics_solution, save_fenics_solution
FLAGS = flags.FLAGS


if __name__ == "__main__":
    flags.DEFINE_float("xmin", 0.0, "scale on random uniform bc")
    flags.DEFINE_float("xmax", 1.0, "scale on random uniform bc")
    flags.DEFINE_float("ymin", -1.0, "scale on random uniform bc")
    flags.DEFINE_float("ymax", 1.0, "scale on random uniform bc")
    flags.DEFINE_integer("max_holes", 12, "scale on random uniform bc")
    flags.DEFINE_float("max_hole_size", 0.4, "scale on random uniform bc")
    FLAGS.ground_truth_resolution = 32


from .td_burgers_common_new import (
    sample_params,
    GroundTruth,
)


def point_theta(theta, c1, c2, x0, y0, size):
    r0 = size * (1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta))
    x = r0 * np.cos(theta)
    y = r0 * np.sin(theta)
    return [x + x0, y + y0]


def solve_fenics(params, boundary_points=24, resolution=16):
    print("solving with params ", params)
    print("resolution ", resolution)
    mesh = fa.IntervalMesh(resolution, FLAGS.xmin,  FLAGS.xmax)

    source_params, ic_params = params
    # pdb.set_trace()
    dt = (FLAGS.tmax - FLAGS.tmin) / (FLAGS.num_tsteps - 1)
    reynolds = fa.Expression("r", degree=1, r=float(1. / source_params[0]))

    V = fa.FunctionSpace(mesh, "CG", 1)

    def left_wall(x, on_boundary):
        return on_boundary and fa.near(x[0], FLAGS.xmin)

    def right_wall(x, on_boundary):
        return on_boundary and fa.near(x[0], FLAGS.xmax)

    loss_fn_lib = importlib.import_module(f'.burgers_formulation.{FLAGS.burgers_pde}', package='src.burgers')

    ic_expression = loss_fn_lib.fa_expressions(params)

    #ic_expression = "cos(2*pi*x[0]) + cos(4*pi*x[0]) + cos(6*pi*x[0])"
    #ic_expression = f"sin(pi*x[0])+" \
    #                f"{float(ic_params[0])}*sin(pi*2.0*x[0])+" \
    #                f"{float(ic_params[1])}*sin(pi*4.0*x[0])"

    u_left = fa.Expression(ic_expression, degree=3)
    u_right = fa.Expression(ic_expression, degree=3)
    bc_left = fa.DirichletBC(V, u_left, left_wall)
    bc_right = fa.DirichletBC(V, u_right, right_wall)
    bc = [bc_left, bc_right]

    u = fa.Function(V)
    v = fa.TestFunction(V)

    u_init = fa.Expression(ic_expression, degree=3)

    u_old = fa.project(u_init, V)
    #u_old = fa.interpolate(u_init, V)

    DT = fa.Constant(dt)
    f = fa.Expression("0.0", degree=0)

    solver_parameters = {
        "newton_solver": {
            "maximum_iterations": FLAGS.max_newton_steps,
            "relaxation_parameter": FLAGS.relaxation_parameter,
            "linear_solver": "mumps",
            "relative_tolerance": 1e-5,
            "absolute_tolerance": 1e-5
        }}

    F = (
        fa.dot(u - u_old, v) / DT + reynolds * fa.inner(fa.grad(u), fa.grad(v)) + fa.inner(u * u.dx(0), v)
        - fa.dot(f, v)
    ) * fa.dx
    #J = fa.derivative(F, u)

    #u = fa.Function(V)

    u_list = []
    t_list = []
    # append initial condition
    u_list.append(u_old.copy(deepcopy=True))
    t_list.append(FLAGS.tmin)
    for n in range(FLAGS.num_tsteps - 1):
        t = FLAGS.tmin + dt * (n + 1)
        try:
            fa.solve(
                F == 0, u, bc,
                solver_parameters=solver_parameters,
            )
        except Exception as e:
            print("Failed solve: ", e)
            print("Failed on params: ", params)
            solver_parameters['newton_solver']['relaxation_parameter'] *= 0.2
            fa.solve(
                F == 0, u, bc,
                solver_parameters=solver_parameters,
            )

        u_list.append(u.copy(deepcopy=True))
        u_old.assign(u)
        t_list.append(t)
    print('Time steps solved by fenics', t_list)

    tmp_filenames = []
    for n in range(FLAGS.num_tsteps):
        u = u_list[n]
        fa.plot(u)
        plt.ylim([-1.2, 3.1])
        plt.grid(True)

        plt.title('Ground Truth \n t = {:.2f}'.format(t_list[n]))
        plt.savefig('td_burger_' + str(n))
        plt.close()
        tmp_filenames.append('td_burger_' + str(n) + '.png')
    build_gif(tmp_filenames)

    #results_array = []
    #x_coord = np.linspace(FLAGS.xmin, FLAGS.xmax, 500)
    #for n in range(FLAGS.num_tsteps):
    #    u = u_list[n]
    #    tmp = []
    #    for x_i in x_coord:
    #        tmp.append(u(x_i))
    #    results_array.append(tmp)
    #results_array = np.array(results_array)
    #plt.imshow(results_array.T, cmap='hot', interpolation='nearest', aspect='auto')
    #plt.show()

    tmp_filenames = []
    for n in range(FLAGS.num_tsteps):
        u = u_list[n]
        fa.plot(u.dx(0))
        plt.grid(True)

        plt.title('Ground Truth \n t = {:.2f}'.format(t_list[n]))
        plt.savefig('td_burger_derivative' + str(n))
        plt.close()
        tmp_filenames.append('td_burger_derivative' + str(n) + '.png')
    build_gif(tmp_filenames, 'td_burger_derivative.gif')

    return GroundTruth(u_list, np.array(t_list))


def build_gif(filenames, outfile=None):
    if outfile is None:
        outfile = 'td_burgers.gif'
    with imageio.get_writer(outfile, mode='I') as writer:
        for f in filenames:
            image = imageio.imread(f)
            writer.append_data(image)
    for f in set(filenames):
        os.remove(f)


def is_defined(xy, u):
    mesh = u.function_space().mesh()
    return (
        mesh.bounding_box_tree().compute_first_entity_collision(fa.Point(xy))
        < mesh.num_cells()
    )


def main(argv):
    params = sample_params(jax.random.PRNGKey(FLAGS.seed))

    print("params: ", params)

    ground_truth = solve_fenics(params, resolution=FLAGS.ground_truth_resolution,
                                boundary_points=int(FLAGS.boundary_resolution_factor*FLAGS.ground_truth_resolution))


if __name__ == "__main__":
    app.run(main)
