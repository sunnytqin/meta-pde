import jax.numpy as np
from absl import app
from absl import flags

FLAGS = flags.FLAGS


# below is for td_burgers_common
def loss_vertical_fn(field_fn, points_on_vertical, params):
    return loss_horizontal_fn(field_fn, points_on_vertical, params)


def loss_horizontal_fn(field_fn, points_on_horizontal, params):
    source_params, bc_params, per_hole_params, n_holes = params

    A0 = (bc_params[0, 0]).astype(float)
    A1 = (bc_params[0, 1]).astype(float)
    A2 = (bc_params[0, 2]).astype(float)
    sinusoidal_magnitude_x = A0 * \
                             np.sin(
                                 A2 * np.pi * (points_on_horizontal[:, 0] - FLAGS.xmin) / (FLAGS.xmax - FLAGS.xmin)) * \
                             np.cos(A2 * np.pi * (points_on_horizontal[:, 1] - FLAGS.ymin) / (FLAGS.ymax - FLAGS.ymin))
    sinusoidal_magnitude_y = A1 * \
                             np.cos(
                                 A2 * np.pi * (points_on_horizontal[:, 0] - FLAGS.xmin) / (FLAGS.xmax - FLAGS.xmin)) * \
                             np.sin(A2 * np.pi * (points_on_horizontal[:, 1] - FLAGS.ymin) / (FLAGS.ymax - FLAGS.ymin))
    # zero_magnitude = np.zeros_like(sinusoidal_magnitude)

    return (field_fn(points_on_horizontal) - np.stack((sinusoidal_magnitude_x, sinusoidal_magnitude_y), axis=-1)) ** 2


def loss_initial_fn(field_fn, points_initial, params):
    source_params, bc_params, per_hole_params, n_holes = params

    A0 = (bc_params[0, 0]).astype(float)
    A1 = (bc_params[0, 1]).astype(float)
    A2 = (bc_params[0, 2]).astype(float)
    sinusoidal_magnitude_x = A0 * \
                             np.sin(A2 * np.pi * (points_initial[:, 0] - FLAGS.xmin) / (FLAGS.xmax - FLAGS.xmin)) * \
                             np.cos(A2 * np.pi * (points_initial[:, 1] - FLAGS.ymin) / (FLAGS.ymax - FLAGS.ymin))
    sinusoidal_magnitude_y = A1 * \
                             np.cos(A2 * np.pi * (points_initial[:, 0] - FLAGS.xmin) / (FLAGS.xmax - FLAGS.xmin)) * \
                             np.sin(A2 * np.pi * (points_initial[:, 1] - FLAGS.ymin) / (FLAGS.ymax - FLAGS.ymin))
    return (field_fn(points_initial) -
            np.stack((sinusoidal_magnitude_x, sinusoidal_magnitude_y), axis=-1)) ** 2

def loss_initial_trivial_fn(field_fn, points_initial, params):
    source_params, bc_params, per_hole_params, n_holes = params

    sinusoidal_magnitude_x = 0.0
    sinusoidal_magnitude_y = 0.0
    return (field_fn(points_initial) -
            np.stack((sinusoidal_magnitude_x, sinusoidal_magnitude_y), axis=-1)) ** 2

# below is for td_burgers_fenics
def fa_expressions():
    initial_condition = ("A0*sin(A2*pi*(x[0]-xmin)/(xmax-xmin))*cos(A2*pi*(x[1]-ymin)/(ymax-ymin))",
                         "A1*cos(A2*pi*(x[0]-xmin)/(xmax-xmin))*sin(A2*pi*(x[1]-ymin)/(ymax-ymin))")
    boundary_condition = initial_condition
    return initial_condition, boundary_condition