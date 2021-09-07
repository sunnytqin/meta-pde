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

if __name__ == "__main__":
    flags.DEFINE_float("xmin", 0.0, "scale on random uniform bc")
    flags.DEFINE_float("xmax", 1.0, "scale on random uniform bc")
    flags.DEFINE_float("ymin", 0.0, "scale on random uniform bc")
    flags.DEFINE_float("ymax", 1.0, "scale on random uniform bc")
    flags.DEFINE_integer("max_holes", 1, "scale on random uniform bc")
    flags.DEFINE_float("max_hole_size", .2, "scale on random uniform bc")
    flags.DEFINE_boolean("stokes_nonlinear", False, "if True, make nonlinear")

from .elasticity_common import (
    plot_solution,
    loss_fn,
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


def solve_fenics(params, boundary_points=32, resolution=32):
    print("solving with params ", params)
    print("resolution ", resolution)
    domain = mshr.Rectangle(
        fa.Point([FLAGS.xmin, FLAGS.ymin]), fa.Point([FLAGS.xmax, FLAGS.ymax]),
    )
    source_params, bc_params, per_hole_params, n_holes = params
    print('per hole params: ', per_hole_params)

    c, xy, size = per_hole_params
    # pdb.set_trace()

    holes = make_domain(c[0], c[1], boundary_points, xy[0], xy[1], size)

    if n_holes > 0:
        obstacle = holes
        domain = domain - obstacle

    mesh = mshr.generate_mesh(domain, resolution)

    def bottom(x, on_boundary):
        return (on_boundary and fa.near(x[1], FLAGS.xmin))

    # Strain function
    def epsilon(u):
        return fa.sym(fa.grad(u))

    # Stress function
    def sigma(u):
        return lambda_ * fa.div(u) * fa.Identity(2) + 2 * mu * epsilon(u)

    # Density
    rho = fa.Constant(1.0)

    # Young's modulus and Poisson's ratio
    E = 200.
    nu = 0.3

    # Lame's constants
    lambda_ = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)

    # Load
    g_x = bc_params[0]
    g_y = bc_params[1]
    b_z = -10.
    g = fa.Constant((g_x, g_y))
    b = fa.Constant((0.0, b_z * rho))

    # Definition of Neumann condition domain
    boundaries = fa.MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)

    top = fa.AutoSubDomain(lambda x: fa.near(x[1], FLAGS.ymax))

    top.mark(boundaries, 1)
    ds = fa.ds(subdomain_data=boundaries)

    V = fa.VectorFunctionSpace(mesh, "CG", 1)
    u = fa.TrialFunction(V)
    v = fa.TestFunction(V)

    bc = fa.DirichletBC(V, fa.Constant((0.0, 0.0)), bottom)

    a = fa.inner(sigma(u), epsilon(v)) * fa.dx
    l = rho * fa.dot(b, v) * fa.dx + fa.inner(g, v) * ds(1)

    u = fa.Function(V)
    A_ass, L_ass = fa.assemble_system(a, l, bc)

    solver_parameters = {
        "newton_solver": {
            "maximum_iterations": 500,
            "linear_solver": "mumps",
            "relaxation_parameter": FLAGS.relaxation_parameter,
        }}

    try:
        fa.solve(a == l, u, bc)
    except Exception as e:
        print("Failed solve: ", e)
        print("Failed on params: ", params)
        solver_parameters['newton_solver']['relaxation_parameter'] *= 0.2
        fa.solve(a == l, u, bc, solver_parameters = solver_parameters)

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

    u = solve_fenics(params)

    x = np.array(u.function_space().tabulate_dof_coordinates()[:100])

    points = sample_points(jax.random.PRNGKey(FLAGS.seed + 1), 128, params)

    #for i in range(3):
    #    solution_dim_i = fa.inner(
    #        u, fa.Constant((0.0 + i == 0, 0.0 + i == 1, 0.0 + i == 2))
    #    )
    #    print(
    #        "norm in dim {}: ".format(i),
    #        fa.assemble(fa.inner(solution_dim_i, solution_dim_i) * fa.dx) / normalizer,
    #    )

    plt.figure(figsize=(9, 9))
    clrs = fa.plot(u, mode='displacement')
    plt.colorbar(clrs)
    plt.show()

if __name__ == "__main__":
    app.run(main)
    args = parser.parse_args()
    args = namedtuple("ArgsTuple", vars(args))(**vars(args))
