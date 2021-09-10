"""
    Solve the hyper elasticity equation using fenics

    PDE: http://www-personal.umich.edu/~chrismav/FEniCS.pdf
    page 14
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


def solve_fenics(params, boundary_points=64, resolution=1):
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
        return on_boundary and fa.near(x[1], FLAGS.xmin)

    # Define Dirichlet boundary (y = 0)
    c = fa.Constant(("0.0", "0.0"))

    V = fa.VectorFunctionSpace(mesh, "Lagrange", 1)

    bc_bottom = fa.DirichletBC(V, c, bottom)
    bcs = [bc_bottom]

    # Define functions
    du = fa.TrialFunction(V)  # Incremental displacement
    v = fa.TestFunction(V)  # Test function
    u = fa.Function(V)  # Displacement from previous iteration
    B = fa.Constant((0.0, -0.5))  # Body force per unit volume
    T = fa.Constant((0.0, 0.0))  # Traction force on the boundary

    # Kinematics
    d = u.geometric_dimension()
    I = fa.Identity(d)  # Identity tensor
    F = I + fa.grad(u)  # Deformation gradient
    C = F.T * F  # Right Cauchy-Green tensor

    # Invariants of deformation tensors
    Ic = fa.tr(C)
    J = fa.det(F)
    Jinv = J ** (-2/d)

    young_mod = 1.0
    poisson_ratio = 0.49

    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))

    # Elasticity parameters
    #E, nu = 10.0, 0.3
    #mu, lmbda = fa.Constant(E / (2 * (1 + nu))), fa.Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))

    # Stored strain energy density (compressible neo-Hookean model)
    psi = (shear_mod / 2) * (Jinv * Ic - d) + (bulk_mod / 2) * (J - 1) ** 2

    # Total potential energy
    Pi = psi * fa.dx - fa.dot(B, u) * fa.dx - fa.dot(T, u) * fa.ds

    # Compute first variation of Pi (directional derivative about u in the direction of v)
    F = fa.derivative(Pi, u, v)

    # Compute Jacobian of F
    J = fa.derivative(F, u, du)

    fa.parameters["form_compiler"]["cpp_optimize"] = True
    ffc_options = {"optimize": True,
                   "eliminate_zeros": True,
                   "precompute_basis_const": True,
                   "precompute_ip_const": True}

    newton_args = {
        "maximum_iterations": 5000,
        "linear_solver": "petsc",
        "relaxation_parameter": 0.001,
        "relative_tolerance": 5e-3,
        "absolute_tolerance": 5e-3
        }
    snes_args = {
        "method": "qn",
        "linear_solver": "petsc",
        "maximum_iterations": 1000,
    }
    solver_args = {
        "nonlinear_solver": "newton",
        "newton_solver": newton_args,
        "snes_solver": snes_args,
    }

    try:
        # Solve variational problem
        fa.solve(F == 0, u, bcs, J=J,
                 form_compiler_parameters=ffc_options,
                 solver_parameters=solver_args)
    except Exception as e:
        print("Failed solve: ", e)
        print("Failed on params: ", params)
        solver_args['newton_solver']['relaxation_parameter'] *= 0.2
        fa.solve(F == 0, u, bcs, J=J,
                 form_compiler_parameters=ffc_options,
                 solver_parameters=solver_args)
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
