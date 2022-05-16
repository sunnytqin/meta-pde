import jax.numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS


# below is for td_burgers_common
def loss_initial_fn(field_fn, points_on_initial, params):
    source_params, ic_params = params

    sinusoidal_magnitude_x = (
            - np.sin(np.pi * points_on_initial[:, 0])
    )

    return (field_fn(points_on_initial) - sinusoidal_magnitude_x) ** 2


def loss_left_fn(field_fn, points_on_left, params):
    return loss_initial_fn(field_fn, points_on_left, params)


def loss_right_fn(field_fn, points_on_right, params):
    return loss_initial_fn(field_fn, points_on_right, params)


# below is for td_burgers_fenics
def fa_expressions(params):
    source_params, ic_params = params
    initial_condition = "-sin(pi*x[0])"
    return initial_condition
