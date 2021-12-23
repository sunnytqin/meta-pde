import jax.numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS


# below is for td_burgers_common
def loss_vertical_fn(field_fn, points_on_vertical, params):
    return loss_horizontal_fn(field_fn, points_on_vertical, params)


def loss_horizontal_fn(field_fn, points_on_horizontal, params):
    source_params, bc_params, per_hole_params, n_holes = params
    bc_x = (points_on_horizontal[:, 0] + points_on_horizontal[:, 1] -
            2 * points_on_horizontal[:, 0] * points_on_horizontal[:, 2]) / (1 - 2 * np.power(points_on_horizontal[:, 2], 2))
    bc_y = (points_on_horizontal[:, 0] - points_on_horizontal[:, 1] -
            2 * points_on_horizontal[:, 1] * points_on_horizontal[:, 2]) / (1 - 2 * np.power(points_on_horizontal[:, 2], 2))

    return (field_fn(points_on_horizontal) - np.stack((bc_x, bc_y), axis=-1)) ** 2


def loss_initial_fn(field_fn, points_initial, params):
    source_params, bc_params, per_hole_params, n_holes = params
    ic_x = points_initial[:, 0] + points_initial[:, 1]
    ic_y = points_initial[:, 0] - points_initial[:, 1]
    return (field_fn(points_initial) - np.stack((ic_x, ic_y), axis=-1)) ** 2


# below is for td_burgers_fenics
def fa_expressions():
    initial_condition = ("x[0]+x[1]", "x[0]-x[1]")
    boundary_condition = ("(x[0]+x[1]-2*x[0]*t)/(1-2*pow(t,2))", "(x[0]-x[1]-2*x[1]*t)/(1-2*pow(t,2))")
    return initial_condition, boundary_condition