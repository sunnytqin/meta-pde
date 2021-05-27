"""
    Solve the linear Stokes equation using fenics

    PDE: https://fenicsproject.org/olddocs/dolfin/1.3.0/python/demo/documented/stokes-iterative/python/documentation.html
"""

import fenics as fa
import matplotlib.pyplot as plt
import mshr
import numpy as np
import pdb
import argparse
import jax
from collections import namedtuple

from absl import app
from absl import flags
from ..util import common_flags

FLAGS = flags.FLAGS

from .burgers_common import (
    plot_solution,
    loss_fn,
    fenics_to_jax,
    SecondOrderTaylorLookup,
    error_on_coords,
    sample_params,
    sample_points,
    is_in_hole,
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

    def inlet(x, on_boundary):
        return on_boundary and fa.near(x[0], FLAGS.xmin)

    def outlet(x, on_boundary):
        return on_boundary and fa.near(x[0], FLAGS.xmax)

    def walls(x, on_boundary):
        return on_boundary and (
            (fa.near(x[1], FLAGS.ymin) or fa.near(x[1], FLAGS.ymax))
            or (not (fa.near(x[0], FLAGS.xmin) or fa.near(x[0], FLAGS.xmax)))
        )

    V = fa.VectorFunctionSpace(mesh, "P", 3)

    u = fa.Function(V)
    v = fa.TestFunction(V)
    du = fa.TrialFunction(V)

    # Define function for setting Dirichlet values
    lhs_expr = fa.Expression(
        ("A0*sin(pi*(x[1]-ymin)/(ymax-ymin))", "A1*sin(pi*(x[1]-ymin)/(ymax-ymin))"),
        A0=float(bc_params[0,0]),
        A1=float(bc_params[0,1]),
        ymax=FLAGS.ymax,
        ymin=FLAGS.ymin,
        element=V.ufl_element(),
    )

    rhs_expr = fa.Expression(
        ("A0*sin(pi*(x[1]-ymin)/(ymax-ymin))","A1*sin(pi*(x[1]-ymin)/(ymax-ymin))"),
        A0=float(bc_params[1,0]),
        A1=float(bc_params[1,1]),
        ymax=FLAGS.ymax,
        ymin=FLAGS.ymin,
        element=V.ufl_element(),
    )
    lhs_u = fa.project(lhs_expr, fa.VectorFunctionSpace(mesh, "P", 2))
    # fa.plot(lhs_u)
    # plt.show()
    # pdb.set_trace()
    bc_in = fa.DirichletBC(V, lhs_expr, inlet)
    bc_out = fa.DirichletBC(V, rhs_expr, outlet)
    bc_walls = fa.DirichletBC(V, fa.Constant((0.0, 0.0)), walls)
    # bc_pressure = fa.DirichletBC(Q, fa.Constant((1.)), inlet)

    dx = fa.Measure("dx")

    # Define variational problem
    u.vector().set_local(np.random.randn(len(u.vector())) * 1e-6)


    #     [ux   uy  ] [u   = [u ux + vuy
    #      vx   vy     v]     u vx + v vy]


    # (u ux - uxx)  +  (v uy - uyy)

    lhs_term = fa.inner(fa.grad(u) * u, v)

    # [ u0xx + u0yy
    #   u1xx + u1yy ]

    #  u0xx * v0 -> - u0x * v0x
    #  u0yy * v0 -> u0y * v0y

    rhs_term = fa.inner(u.dx(0), v.dx(0)) + fa.inner(u.dx(1), v.dx(1))

    # rhs_term = fa.inner((u.dx(0).dx(0) + u.dx(1).dx(1)), v)

    F = (
        lhs_term * fa.dx - rhs_term * fa.dx
    )
    solver_parameters = {
        "newton_solver": {
            "maximum_iterations": FLAGS.max_newton_steps,
            "relaxation_parameter": FLAGS.relaxation_parameter,
            "linear_solver": "mumps",
        }}

    try:
        fa.solve(
            F == 0,
            u,
            [bc_walls, bc_in, bc_out],
            solver_parameters=solver_parameters,
        )
    except Exception as e:
        print("Failed solve: ", e)
        print("Failed on params: ", params)
        solver_parameters['newton_solver']['relaxation_parameter'] *= 0.2
        fa.solve(F==0, u, [bc_walls, bc_in, bc_out], solver_parameters=solver_parameters)

    return u


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

    u = solve_fenics(params, resolution=FLAGS.ground_truth_resolution,
                     boundary_points=int(FLAGS.boundary_resolution_factor*FLAGS.ground_truth_resolution))

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
    plot_solution(u, params)
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

    plt.scatter(points_domain[:, 0], points_domain[:, 1], color="b", alpha=0.5)

    plt.show()


if __name__ == "__main__":
    app.run(main)
    args = parser.parse_args()
    args = namedtuple("ArgsTuple", vars(args))(**vars(args))
