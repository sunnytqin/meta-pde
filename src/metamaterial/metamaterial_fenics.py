import argparse
import fenics as fa
from .mm_pde.metamaterial import Metamaterial
from .metamaterial_common import *

import numpy as npo
import copy

import jax.numpy as np
import jax
from functools import partial

import matplotlib.pyplot as plt

from ..util.timer import Timer

import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--L0", help="length between pore centers", type=float, default=2.0)
parser.add_argument(
    "--porosity", help="% material removed for pore, in [0, 1]", type=float, default=0.5
)
parser.add_argument(
    "--c1", help="low-freq param for pore shape", type=float, default=-0.1
)
parser.add_argument(
    "--c2", help="high-freq param for pore shape", type=float, default=0.1
)
parser.add_argument(
    "--metamaterial_mesh_size",
    help="finite elements along one edge of cell; "
    " Overvelde&Bertoldi use about sqrt(1000)",
    type=int,
    default=30,
)
parser.add_argument(
    "--pore_radial_resolution",
    help="num points used to define geometry of pore boundary",
    type=int,
    default=60,
)
parser.add_argument(
    "--n_cells", help="number cells on one side of ref volume", type=int, default=1
)
parser.add_argument(
    "--young_modulus", help="young's modulus of base material", type=float, default=1.0
)
parser.add_argument(
    "--poisson_ratio", help="poisson's ratio of base material", type=float, default=0.49
)
parser.add_argument(
    "--relaxation_parameter",
    default=0.9,
    type=float,
    help="relaxation parameter for Newton",
)
parser.add_argument(
    "--nonlinear_solver",
    default="newton",
    type=str,
    help="Nonlinear solver: newton or snes",
)
parser.add_argument(
    "--snes_method", default="qn", type=str, help="newtontr, newtonls, qn, ..."
)
parser.add_argument(
    "--linear_solver", default="petsc", type=str, help="Newton linear solver"
)
parser.add_argument("--preconditioner", default="ilu", type=str, help="Preconditioner")
parser.add_argument(
    "--max_newton_iter", default=25, type=int, help="Newton maximum iters"
)
parser.add_argument(
    "--max_snes_iter", default=1000, type=int, help="Newton maximum iters"
)
parser.add_argument(
    "--adaptive", default=False, action="store_true", help="Use adaptive solver"
)
parser.add_argument(
    "--manual_solver",
    default=False,
    action="store_true",
    help="Use homemade Newton solver",
)
parser.add_argument(
    "--min_feature_size",
    help="minimum distance between pore boundaries = minimum "
    "width of a ligament / material section in structure. We "
    "also use this as minimum width of pore.",
    type=float,
    default=0.15,
)


def solve_fenics(source_params, bc_params, geo_params):
    args, _ = parser.parse_known_args()

    if geo_params is not None:
        c1, c2 = geo_params
        args.c1 = c1
        args.c2 = c2

    if source_params is not None:
        young_mod, poisson_ratio = source_params
        args.young_mod = young_mod
        args.poisson_ratio = poisson_ratio

    with Timer() as t:
        mm = Metamaterial(args)
    print("made metamaterial in {}s".format(t.interval))

    with Timer() as t:
        if bc_params is not None:

            dofs = mm.V.tabulate_dof_coordinates().reshape(-1, 2, 2)[:, 0, :]
            outputs = vmap_boundary_conditions(dofs, bc_params)
            bc_vec = outputs.reshape(-1)

            bc_V = fa.Function(mm.V)
            assert len(bc_vec) == len(bc_V.vector())
            bc_V.vector().set_local(bc_vec)

        else:
            bc_V = fa.Function(mm.V)

    print("made bc in {}s".format(t.interval))

    with Timer() as t:
        u = solve_recurse(args, mm, bc_V, 0, None)
    print("solved in {}s".format(t.interval))

    return u


def make_fenics(field_fn, geo_params):
    args, _ = parser.parse_known_args()
    if geo_params is not None:
        c1, c2 = geo_params
        args.c1 = c1
        args.c2 = c2
    mm = Metamaterial(args)

    dofs = mm.V.tabulate_dof_coordinates().reshape(-1, 2, 2)[:, 0, :]
    outputs = field_fn(dofs)
    field_vec = outputs.reshape(-1).astype(np.float32)

    field_V = fa.Function(mm.V)
    assert len(field_vec) == len(field_V.vector())
    field_V.vector().set_local(field_vec)

    return field_V


def solve_recurse(args, mm, bc_V, iter, guess=None):
    if guess is None:
        guess = npo.random.randn(len(fa.Function(mm.V).vector()[:])) * 1e-14
    try:
        u = mm.solve_problem(args, boundary_fn=bc_V, initial_guess=guess)
        return u
    except Exception as e:
        print(
            "failed trying iter {} with relax {} max_newton {}".format(
                iter, args.relaxation_parameter, args.max_newton_iter
            )
        )
        if iter > 50:
            raise (e)
        args = copy.deepcopy(args)
        args.relaxation_parameter = args.relaxation_parameter * 0.5
        args.max_newton_iter = int(args.max_newton_iter / 0.5) + 1

        intermediate_bc = fa.project(bc_V * 0.5, mm.V)
        intermediate_u = solve_recurse(args, mm, intermediate_bc, iter + 1, guess)
        intermediate_guess = intermediate_u.vector()
        return solve_recurse(args, mm, bc_V, iter + 1, intermediate_guess)


if __name__ == "__main__":
    # args = parser.parse_args()

    # source_params = np.random.randn(3, 4)
    bc_params = 0.1 * np.random.randn(2, 5)

    u = solve_fenics(None, bc_params, None)
    fa.plot(u, mode="displacement", title="solution")
    plt.show()
