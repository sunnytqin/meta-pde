import jax.numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS


# below is for td_burgers_common
def loss_initial_fn(field_fn, points_on_initial, params):
    source_params, ic_params = params

    # ic_expression = ic_expression = "sin(pi*x[0])+3.0*sin(pi*2.0*x[0])-4.0*sin(pi*4.0*x[0])+5.0*sin(pi*6.0*x[0])"
    #sinusoidal_magnitude_x = (
    #        np.sin(np.pi * points_on_initial[:, 0]) + 3.0 * np.sin(2.0 * np.pi * points_on_initial[:, 0]) -
    #        4.0 * np.sin(4.0 * np.pi * points_on_initial[:, 0]) + 5.0 * np.sin(6.0 * np.pi * points_on_initial[:, 0])
    #)
    sinusoidal_magnitude_x = (
            np.sin(np.pi * points_on_initial[:, 0]) +
            ic_params[0] * np.sin(2.0 * np.pi * points_on_initial[:, 0]) +
            ic_params[1] * np.sin(4.0 * np.pi * points_on_initial[:, 0])
    )

    return (field_fn(points_on_initial) - sinusoidal_magnitude_x) ** 2


def loss_left_fn(field_fn, points_on_left, params):
    return loss_initial_fn(field_fn, points_on_left, params)


def loss_right_fn(field_fn, points_on_right, params):
    return loss_initial_fn(field_fn, points_on_right, params)


# below is for td_burgers_fenics
def fa_expressions(params):
    source_params, ic_params = params
    initial_condition = f"sin(pi*x[0])+" \
                        f"{float(ic_params[0])}*sin(pi*2.0*x[0])+" \
                        f"{float(ic_params[1])}*sin(pi*4.0*x[0])"

    return initial_condition