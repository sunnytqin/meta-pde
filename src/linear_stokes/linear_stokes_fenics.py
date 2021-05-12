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

from .linear_stokes_common import (
    plot_solution,
    loss_fn,
    fenics_to_jax,
    SecondOrderTaylorLookup,
    error_on_coords,
    sample_params,
    sample_points,
    is_in_hole,
    XMIN,
    XMAX,
    YMIN,
    YMAX,
    PRESSURE_FACTOR,
)


parser = argparse.ArgumentParser()
parser.add_argument("--vary_source", type=int, default=1, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=1, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=1, help="1=true.")
parser.add_argument("--bc_scale", type=float, default=1e0, help="bc scale")
parser.add_argument("--seed", type=int, default=0, help="set random seed")


def point_theta(theta, c1, c2, x0, y0, size):
    r0 = size * (1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta))
    x = r0 * np.cos(theta)
    y = r0 * np.sin(theta)
    return fa.Point(np.array([x + x0, y + y0]))


def make_domain(c1, c2, n_points, x0, y0, size):
    thetas = np.linspace(0.0, 1.0, n_points) * 2 * np.pi
    points = [point_theta(t, c1, c2, x0, y0, size) for t in thetas]
    return mshr.Polygon(points)


def solve_fenics(params, boundary_points=32, resolution=32):
    domain = mshr.Rectangle(
        fa.Point(np.array([XMIN, YMIN])), fa.Point(np.array([XMAX, YMAX]))
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
        return on_boundary and fa.near(x[0], XMIN)

    def outlet(x, on_boundary):
        return on_boundary and fa.near(x[0], XMAX)

    def walls(x, on_boundary):
        return on_boundary and not (fa.near(x[0], XMIN) or fa.near(x[0], XMAX))

    V_h = fa.VectorElement("CG", mesh.ufl_cell(), 2)
    Q_h = fa.FiniteElement("CG", mesh.ufl_cell(), 1)
    W = fa.FunctionSpace(mesh, V_h * Q_h)
    V, Q = W.split()
    u_p = fa.Function(W)
    u, p = fa.split(u_p)
    v_q = fa.TestFunction(W)
    v, q = fa.split(v_q)
    du_p = fa.TrialFunction(W)

    # Define function for setting Dirichlet values
    lhs_expr = fa.Expression(
        ("A*sin(pi*(x[1]-YMIN)/(YMAX-YMIN))", 0.0),
        A=float(bc_params[0]),
        YMAX=YMAX,
        YMIN=YMIN,
        element=V.ufl_element(),
    )
    lhs_u = fa.project(lhs_expr, fa.VectorFunctionSpace(mesh, "P", 2))
    # fa.plot(lhs_u)
    # plt.show()
    # pdb.set_trace()
    bc_in = fa.DirichletBC(V, lhs_u, inlet)
    bc_out = fa.DirichletBC(Q, fa.Constant((0.0)), outlet)
    bc_walls = fa.DirichletBC(V, fa.Constant((0.0, 0.0)), walls)
    # bc_pressure = fa.DirichletBC(Q, fa.Constant((1.)), inlet)

    dx = fa.Measure("dx")

    # Define variational problem
    u_p.vector().set_local(np.random.randn(len(u_p.vector())) * 1e-6)

    F = (
        fa.inner(fa.grad(u), fa.grad(v)) * fa.dx
        + PRESSURE_FACTOR
        * p
        * fa.div(v)
        * fa.dx  # Pressure varies on 100x scale of velocity
        + q * fa.div(u) * fa.dx
    )

    fa.solve(
        F == 0,
        u_p,
        [bc_walls, bc_in, bc_out],
        solver_parameters={
            "newton_solver": {
                "maximum_iterations": int(1000),
                "relaxation_parameter": 0.3,
                "linear_solver": "mumps",
            }
        },
    )

    return u_p


def is_defined(xy, u):
    mesh = u.function_space().mesh()
    return (
        mesh.bounding_box_tree().compute_first_entity_collision(fa.Point(xy))
        < mesh.num_cells()
    )


if __name__ == "__main__":
    args = parser.parse_args()
    args = namedtuple("ArgsTuple", vars(args))(**vars(args))

    params = sample_params(jax.random.PRNGKey(args.seed), args)
    source_params, bc_params, per_hole_params, num_holes = params

    print("params: ", params)

    u_p = solve_fenics(params)

    x = np.array(u_p.function_space().tabulate_dof_coordinates()[:100])

    points = sample_points(jax.random.PRNGKey(args.seed + 1), 128, params)

    normalizer = fa.assemble(
        fa.project(
            fa.Constant((1.0)), fa.FunctionSpace(u_p.function_space().mesh(), "P", 2)
        )
        * fa.dx
    )
    for i in range(3):
        solution_dim_i = fa.inner(
            u_p, fa.Constant((0.0 + i == 0, 0.0 + i == 1, 0.0 + i == 2))
        )
        print(
            "norm in dim {}: ".format(i),
            fa.assemble(fa.inner(solution_dim_i, solution_dim_i) * fa.dx) / normalizer,
        )

    u, p = u_p.split()
    plt.figure(figsize=(9, 3))
    clrs = fa.plot(p)
    plt.colorbar(clrs)
    plt.show()

    plot_solution(u_p, params)
    plt.show()

    u, p = u_p.split()
    # plot solution
    X, Y = np.meshgrid(np.linspace(XMIN, XMAX, 300), np.linspace(YMIN, YMAX, 100))
    Xflat, Yflat = X.reshape(-1), Y.reshape(-1)

    # X, Y = X[valid], Y[valid]
    valid = [is_defined([x, y], u_p) for x, y in zip(Xflat, Yflat)]

    UV = [
        u_p(x, y)[:2] if is_defined([x, y], u_p) else np.array([0.0, 0.0])
        for x, y in zip(Xflat, Yflat)
    ]

    U = np.array([uv[0] for uv in UV]).reshape(X.shape)
    V = np.array([uv[1] for uv in UV]).reshape(Y.shape)

    X_, Y_ = np.meshgrid(np.linspace(XMIN, XMAX, 60), np.linspace(YMIN, YMAX, 40))
    Xflat_, Yflat_ = X_.reshape(-1), Y_.reshape(-1)

    # X, Y = X[valid], Y[valid]
    valid_ = [is_defined([x, y], u_p) for x, y in zip(Xflat_, Yflat_)]
    Xflat_, Yflat_ = Xflat_[valid_], Yflat_[valid_]
    UV_ = [u_p(x, y)[:2] for x, y in zip(Xflat_, Yflat_)]

    U_ = np.array([uv[0] for uv in UV_])
    V_ = np.array([uv[1] for uv in UV_])

    plt.figure(figsize=(9, 3))
    fa.plot(p)
    plt.show()

    plt.figure(figsize=(9, 3))
    parr = np.array([p([x, y]) for x, y in zip(Xflat_, Yflat_)])
    fa.plot(
        p,
        mode="color",
        shading="gouraud",
        edgecolors="k",
        linewidth=0.0,
        cmap="BuPu",
        vmin=parr.min() - 0.5 * parr.max() - 0.5,
        vmax=2 * parr.max() - parr.min() + 0.5,
    )

    speed = np.linalg.norm(np.stack([U, V], axis=2), axis=2)

    speed_ = np.linalg.norm(np.stack([U_, V_], axis=1), axis=1)

    seed_points = np.stack(
        [XMIN * np.ones(40), np.linspace(YMIN + 0.1, YMAX - 0.1, 40)], axis=1
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
    # plt.scatter(X_[in_hole], Y_[in_hole], c='gray', s=0.1)

    # plt.figure(figsize=(9, 3))

    plt.figure(figsize=(9, 3))
    fa.plot(u_p.function_space().mesh())

    plt.figure(figsize=(9, 3))
    (
        points_inlet,
        points_outlet,
        points_wall,
        points_pores,
        points_domain,
    ) = sample_points(jax.random.PRNGKey(args.seed + 1), 1024, params)

    points_wall = np.concatenate([points_wall, points_pores])

    plt.scatter(points_wall[:, 0], points_wall[:, 1], color="g", alpha=0.5)
    plt.scatter(points_inlet[:, 0], points_inlet[:, 1], color="r", alpha=0.5)
    plt.scatter(points_outlet[:, 0], points_outlet[:, 1], color="y", alpha=0.5)

    plt.scatter(points_domain[:, 0], points_domain[:, 1], color="b", alpha=0.5)

    plt.show()
