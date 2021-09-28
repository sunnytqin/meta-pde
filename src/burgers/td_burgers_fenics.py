"""
    Solve the burger's equation using fenics

    PDE: https://www.sciencedirect.com/science/article/pii/S0021999110003190
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

from absl import app
from absl import flags
from ..util import common_flags
from ..util.trainer_util import read_fenics_solution, save_fenics_solution

FLAGS = flags.FLAGS

if __name__ == "__main__":
    flags.DEFINE_float("xmin", -1.0, "scale on random uniform bc")
    flags.DEFINE_float("xmax", 1.0, "scale on random uniform bc")
    flags.DEFINE_float("ymin", -1.0, "scale on random uniform bc")
    flags.DEFINE_float("ymax", 1.0, "scale on random uniform bc")
    flags.DEFINE_integer("max_holes", 0, "scale on random uniform bc")
    flags.DEFINE_float("max_hole_size", 0.4, "scale on random uniform bc")
    FLAGS.ground_truth_resolution = 16

from .td_burgers_common import (
    plot_solution,
    loss_fn,
    SecondOrderTaylorLookup,
    error_on_coords,
    sample_params,
    sample_points,
    GroundTruth,
)


def point_theta(theta, c1, c2, x0, y0, size):
    r0 = size * (1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta))
    x = r0 * np.cos(theta)
    y = r0 * np.sin(theta)
    return [x + x0, y + y0]


def make_domain(c1, c2, n_points, x0, y0, size):
    try:
        thetas = np.linspace(0.0, 1.0, n_points, endpoint=False) * 2 * np.pi
        points = [point_theta(t, c1, c2, x0, y0, size) for t in thetas]
        # print(points)
        domain = mshr.Polygon([fa.Point(p) for p in points])
        return domain
    except Exception as e:
        print("error constructing domain")
        print("params: ", (c1, c2, n_points, x0, y0, size))
        pdb.set_trace()


def solve_fenics(params, boundary_points=24, resolution=16):
    print("solving with params ", params)
    print("resolution ", resolution)
    domain = mshr.Rectangle(
        fa.Point([FLAGS.xmin, FLAGS.ymin]), fa.Point([FLAGS.xmax, FLAGS.ymax]),
    )
    source_params, bc_params, per_hole_params, n_holes = params
    # pdb.set_trace()
    dt = (FLAGS.tmax - FLAGS.tmin) / FLAGS.num_tsteps

    holes = [
        make_domain(c1, c2, boundary_points, x0, y0, size)
        for c1, c2, x0, y0, size in per_hole_params[:n_holes]
    ]

    if n_holes > 0:
        obstacle = holes[0]
        for o2 in holes[1:]:
            obstacle = obstacle + o2
        domain = domain - obstacle

    mesh = mshr.generate_mesh(domain, resolution)

    def vertical_walls(x, on_boundary):
        return on_boundary and (fa.near(x[0], FLAGS.xmin) or fa.near(x[0], FLAGS.xmax))

    def left_wall(x, on_boundary):
        return on_boundary and fa.near(x[0], FLAGS.xmin)

    def right_wall(x, on_boundary):
        return on_boundary and fa.near(x[0], FLAGS.xmax)

    def horizontal_walls(x, on_boundary):
        return on_boundary and (fa.near(x[1], FLAGS.ymin) or fa.near(x[1], FLAGS.ymax))

    def non_slip_walls(x, on_boundary):
        return on_boundary and (
            (not (fa.near(x[0], FLAGS.xmin) or fa.near(x[0], FLAGS.xmax)))
            and
            (not (fa.near(x[1], FLAGS.ymin) or fa.near(x[1], FLAGS.ymax)))
        )


    V = fa.VectorFunctionSpace(mesh, "P", 3)

    u = fa.Function(V)
    v = fa.TestFunction(V)

    reuse = True
    if reuse:
        # generate cache data
        cache = {
            "hparams": (resolution,
                        FLAGS.xmin, FLAGS.xmax,
                        FLAGS.ymin, FLAGS.ymax,
                        FLAGS.tmin, FLAGS.tmax, FLAGS.num_tsteps,
                        ),
            "params": params,
            "pde": "td_burgers"
        }
        # check if PDE has already been solved

        u_list = []
        t_list = []
        for n in range(FLAGS.num_tsteps):
            u_list.append(u.copy(deepcopy=True))
            t_list.append(FLAGS.tmin + dt * n)
        solved = read_fenics_solution(cache, u_list)
        if solved and reuse:
            return GroundTruth(u_list, np.array(t_list))

    # Define function for setting Dirichlet values
    non_slip = fa.Constant((0.0, 0.0))

    u_D = fa.Expression(
        ("A0*sin(A2*pi*(x[0]-xmin)/(xmax-xmin))*cos(A3*pi*(x[1]-ymin)/(ymax-ymin))",
         "A1*cos(A2*pi*(x[0]-xmin)/(xmax-xmin))*sin(A3*pi*(x[1]-ymin)/(ymax-ymin))"),
        A0=float(bc_params[0, 0]),
        A1=float(bc_params[0, 1]),
        A2=float(bc_params[0, 2]),
        A3=float(bc_params[0, 3]),
        xmax=FLAGS.xmax,
        xmin=FLAGS.xmin,
        ymax=FLAGS.ymax,
        ymin=FLAGS.ymin,
        element=V.ufl_element())
    u_n = fa.project(u_D, V)

    # Define variational problem
    u.vector().set_local(np.random.randn(len(u.vector())) * 1e-6)

    solver_parameters = {
        "newton_solver": {
            "maximum_iterations": FLAGS.max_newton_steps,
            "relaxation_parameter": FLAGS.relaxation_parameter,
            "linear_solver": "mumps",
            #"relative_tolerance": 1e-8,
            #"absolute_tolerance": 1e-8
        }}

    tmp_filenames = []
    u_list = []
    t_list = []
    # append intial condition
    u_list.append(u_n.copy(deepcopy=True))
    t_list.append(FLAGS.tmin)
    for n in range(FLAGS.num_tsteps - 1):
        t = FLAGS.tmin + dt * (n + 1)

        left_expr = fa.Expression(
            ("0.0",
             "A1*cos(A2*pi*(x[0]-xmin)/(xmax-xmin))*sin(A3*pi*(x[1]-ymin)/(ymax-ymin))"),
            A0=float(bc_params[0, 0]),
            A1=float(bc_params[0, 1]),
            A2=float(bc_params[0, 2]),
            A3=float(bc_params[0, 3]),
            xmax=FLAGS.xmax,
            xmin=FLAGS.xmin,
            ymax=FLAGS.ymax,
            ymin=FLAGS.ymin,
            element=V.ufl_element(),
        )

        right_expr = fa.Expression(
            ("0.0",
             "A1*cos(A2*pi*(x[0]-xmin)/(xmax-xmin))*sin(A3*pi*(x[1]-ymin)/(ymax-ymin))"),
            A0=float(bc_params[0, 0]),
            A1=float(bc_params[0, 1]),
            A2=float(bc_params[0, 2]),
            A3=float(bc_params[0, 3]),
            xmax=FLAGS.xmax,
            xmin=FLAGS.xmin,
            ymax=FLAGS.ymax,
            ymin=FLAGS.ymin,
            element=V.ufl_element(),
        )

        horizontal_expr = fa.Expression(
            ("A0*sin(A2*pi*(x[0]-xmin)/(xmax-xmin))*cos(A3*pi*(x[1]-ymin)/(ymax-ymin))",
             "0.0"),
            A0=float(bc_params[0, 0]),
            A1=float(bc_params[0, 1]),
            A2=float(bc_params[0, 2]),
            A3=float(bc_params[0, 3]),
            xmax=FLAGS.xmax,
            xmin=FLAGS.xmin,
            ymax=FLAGS.ymax,
            ymin=FLAGS.ymin,
            t=t,
            element=V.ufl_element(),
        )
        bc_left = fa.DirichletBC(V, left_expr, left_wall)
        bc_right = fa.DirichletBC(V, right_expr, right_wall)
        bc_horizontal = fa.DirichletBC(V, horizontal_expr, horizontal_walls)
        bc_nonslip = fa.DirichletBC(V, non_slip, non_slip_walls)

        #     [ux   uy  ] [u   = [u ux + vuy
        #      vx   vy     v]     u vx + v vy]

        # (u ux - uxx)  +  (v uy - uyy)
        #reynolds = (2*np.exp(t) + 1) * float(1 / source_params[0])
        #p1 = fa.Expression("r", degree=1, r=float(1. / source_params[0]))
        #p2 = fa.Expression("0.3*r", degree=1, r=float(1. / source_params[0]))
        #reynolds = fa.Expression("pow(x[0],2)+pow(x[1],2)+2.*x[0]*x[1] < pow(0.8,2) + DOLFIN_EPS ? p1 : p2",
        #                  p1=p1,
        #                  p2=p2,
        #                  degree=2)

        #reynolds = fa.Expression("(sin(pi*x[0])+1.0)*r",
        #                         r=float(1 / source_params[0]),
        #                         degree=2
        #                         )

        reynolds = fa.Expression("r", degree=1, r=float(1. / source_params[0]))

        dt = (FLAGS.tmax - FLAGS.tmin) / FLAGS.num_tsteps

        lhs_term = fa.dot(u, v) + \
                   reynolds * dt * fa.inner(fa.grad(u), fa.grad(v)) + \
                   dt * fa.inner(fa.grad(u) * u, v)

        # [ u0xx + u0yy
        #   u1xx + u1yy ]

        #  u0xx * v0 -> - u0x * v0x
        #  u0yy * v0 -> u0y * v0y

        rhs_term = fa.dot(u_n, v)

        F = (
                lhs_term * fa.dx - rhs_term * fa.dx
        )


        try:
            fa.solve(
                F == 0,
                u,
                [bc_horizontal, bc_left, bc_right],
                solver_parameters=solver_parameters,
            )
        except Exception as e:
            print("Failed solve: ", e)
            print("Failed on params: ", params)
            solver_parameters['newton_solver']['relaxation_parameter'] *= 0.2
            fa.solve(F == 0, u, [bc_left, bc_right, bc_horizontal], solver_parameters=solver_parameters)

        u_list.append(u.copy(deepcopy=True))
        u_n.assign(u)
        t_list.append(t)

    for n in range(FLAGS.num_tsteps):
        u = u_list[n]
        intensity = fa.inner(u, u)

        plt.figure(figsize=(5, 5))
        fa.plot(intensity,
                mode="color",
                shading="gouraud",
                edgecolors="k",
                linewidth=0.0,
                cmap="BuPu",
                )
        fa.plot(u)
        plt.title('Ground Truth \n t = {:.2f}'.format(t_list[n]))
        plt.savefig('td_burger_' + str(n))
        plt.close()
        tmp_filenames.append('td_burger_' + str(n) + '.png')

    print('Time steps solved by fenics', t_list)

    if reuse:
        path = save_fenics_solution(cache, GroundTruth(u_list, np.array(t_list)))
        build_gif(tmp_filenames, os.path.join(path, 'td_burgers.gif'))
    else:
        build_gif(tmp_filenames)

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
    source_params, bc_params, per_hole_params, num_holes = params

    print("params: ", params)

    ground_truth = solve_fenics(params, resolution=FLAGS.ground_truth_resolution,
                     boundary_points=int(FLAGS.boundary_resolution_factor*FLAGS.ground_truth_resolution))

    u = ground_truth[-1]
    x = np.array(u.function_space().tabulate_dof_coordinates()[:100])

    points = sample_points(jax.random.PRNGKey(FLAGS.seed + 1), 128, params)

    normalizer = fa.assemble(
        fa.project(
            fa.Constant((1.0)), fa.FunctionSpace(u.function_space().mesh(), "P", 2)
        )
        * fa.dx
    )
    for i in range(2):
        solution_dim_i = fa.inner(
            u, fa.Constant((0.0 + i == 0, 0.0 + i == 1))
        )
        print(
            "norm in dim {}: ".format(i),
            fa.assemble(fa.inner(solution_dim_i, solution_dim_i) * fa.dx) / normalizer,
        )

    plt.figure(figsize=(5, 5))
    #plot_solution(u, params)
    fa.plot(u)
    plt.show()

    plt.figure(figsize=(5, 5))
    fa.plot(u.function_space().mesh())

    plt.figure(figsize=(5, 5))
    (
        points_inlet,
        points_outlet,
        points_wall,
        points_pores,
        points_domain,
    ) = sample_points(jax.random.PRNGKey(FLAGS.seed + 1), 1024, params)

    points_wall = np.concatenate([points_wall, points_pores])

    plt.scatter(points_wall[:, 0], points_wall[:, 1], color="g", alpha=0.5)
    plt.scatter(points_inlet[:, 0], points_inlet[:, 1], color="r", alpha=0.5)
    plt.scatter(points_outlet[:, 0], points_outlet[:, 1], color="y", alpha=0.5)
    plt.scatter(points_pores[:, 0], points_pores[:, 1], color="k", alpha=0.5)
    plt.scatter(points_domain[:, 0], points_domain[:, 1], color="b", alpha=0.5)

    plt.show()


if __name__ == "__main__":
    #flags.DEFINE_float("xmin", -1.0, "scale on random uniform bc")
    #flags.DEFINE_float("xmax", 1.0, "scale on random uniform bc")
    #flags.DEFINE_float("ymin", -1.0, "scale on random uniform bc")
    #flags.DEFINE_float("ymax", 1.0, "scale on random uniform bc")
    #flags.DEFINE_integer("max_holes", 0, "scale on random uniform bc")
    #flags.DEFINE_float("max_hole_size", 0.4, "scale on random uniform bc")
    #FLAGS.ground_truth_resolution = 32
    app.run(main)
    #args = parser.parse_args()
    #args = namedtuple("ArgsTuple", vars(args))(**vars(args))
