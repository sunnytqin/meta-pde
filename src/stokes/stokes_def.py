"""A container for PDE-specific attributes.

Stuff exposed via importing this module should have the same names as stuff exposed
by importing any other pde_def"""

from ..nets.field import make_nf_ndim, DivFreeVelocityPressureField

from .stokes_common import (
    sample_params,
    sample_points,
    loss_fn,
    plot_solution,
    sample_points_in_domain,
)

from .stokes_fenics import solve_fenics

dim = 3

# BaseField = make_nf_ndim(3)
BaseField = DivFreeVelocityPressureField
