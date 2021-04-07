from ..nets.field import NeuralPotential

from .poisson_common import (
    sample_params,
    sample_points,
    sample_points_in_domain,
    loss_fn,
    plot_solution,
)

from .poisson_fenics import solve_fenics

dim = 1

BaseField = NeuralPotential
