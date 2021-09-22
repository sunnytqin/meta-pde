"""A container for PDE-specific attributes.

Stuff exposed via importing this module should have the same names as stuff exposed
by importing any other pde_def"""

from ..nets.field import make_nf_ndim, DivFreeVelocityPressureField, make_res_nf_ndim

from .td_burgers_common import (
    sample_params,
    sample_points,
    loss_fn,
    plot_solution,
    sample_points_in_domain,
    sample_points_initial,
)
from .td_burgers_fenics import build_gif

from .td_burgers_fenics import (solve_fenics, GroundTruth)

dim = 2

BaseField = make_nf_ndim(dim)
#BaseField = make_res_nf_ndim(dim)
