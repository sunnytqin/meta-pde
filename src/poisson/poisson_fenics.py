"""
    Solve a nonlinear Poisson equation using Fenics,

    nabla dot ((1 + 0.1 u**2) nabla u) = f(x)

    per https://fenicsproject.org/olddocs/dolfin/1.4.0/python/demo/documented/nonlinear-poisson/python/documentation.html

    where u is the potential we are solving for, f is a source term,
    and the laplacian is the trace of the hessian (d^2 u / dx^2).

    todo -- consider changing the 0.1 to also be a sampled parameter
"""

import numpy as np
import mshr
import jax

import time

import contextlib

import matplotlib.pyplot as plt
import pdb

import fenics as fa

import argparse
from collections import namedtuple

from .poisson_common import sample_params, boundary_conditions, source

parser = argparse.ArgumentParser()
parser.add_argument("--vary_source", type=int, default=0, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=0, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=0, help="1=true.")
parser.add_argument("--bc_scale", type=float, default=1e-2, help="bc scale")


def point_theta(theta, c1, c2):
    r0 = 1.0 + c1 * np.cos(4 * theta) + c2 * np.cos(8 * theta)
    x = r0 * np.cos(theta)
    y = r0 * np.sin(theta)
    return fa.Point(np.array([x, y]))


def make_domain(c1, c2, n_points):
    thetas = np.linspace(0.0, 1.0, n_points) * 2 * np.pi
    points = [point_theta(t, c1, c2) for t in thetas]
    return mshr.Polygon(points)


def solve_fenics(params, boundary_points=64, resolution=12):
    source_params, bc_params, geo_params = params
    c1, c2 = geo_params
    domain = make_domain(c1, c2, boundary_points)
    mesh = mshr.generate_mesh(domain, resolution)
    V = fa.FunctionSpace(mesh, "P", 2)

    class Boundary(fa.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    class BCExpression(fa.UserExpression):
        def eval(self, value, x):
            value[0] = np.array(boundary_conditions(bc_params, x))

        def value_shape(self):
            return ()

    class SourceExpression(fa.UserExpression):
        def eval(self, value, x):
            value[0] = np.array(source(source_params, x))

        def value_shape(self):
            return ()

    u = fa.Function(V)
    v = fa.TestFunction(V)
    source_V = fa.interpolate(SourceExpression(), V)
    bc_V = fa.interpolate(BCExpression(), V)

    bc = fa.DirichletBC(V, bc_V, Boundary())

    F = (
        fa.inner((1 + 1e-1 * u ** 2) * fa.grad(u), fa.grad(v)) * fa.dx
        + source_V * v * fa.dx
    )
    fa.solve(F == 0, u, bc)

    return u


if __name__ == "__main__":
    c1 = -0.1
    c2 = 0.1
    args = parser.parse_args()
    args = namedtuple("ArgsTuple", vars(args))(**vars(args))
    params = sample_params(jax.random.PRNGKey(0), args)

    u = solve_fenics(params)
    fa.plot(u, title="solution")
    plt.show()
