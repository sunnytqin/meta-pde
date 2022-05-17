"""A container for PDE-specific attributes.

Stuff exposed via importing this module should have the same names as stuff exposed
by importing any other pde_def"""

from ..nets.field import make_nf_ndim, NeuralPotential

from .td_burgers_common_new import (
    sample_params,
    sample_points,
    loss_fn,
    plot_solution,
    sample_points_in_domain,
    sample_points_initial,
)

from .td_burgers_fenics_new import (solve_fenics, GroundTruth)


dim = 1

BaseField = NeuralPotential
#BaseField = make_nf_ndim(dim)
