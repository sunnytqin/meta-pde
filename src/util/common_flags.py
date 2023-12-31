import absl
from absl import app
from absl import flags

from jax.config import config
config.update("jax_log_compiles", 0)

FLAGS = flags.FLAGS

# Logging
flags.DEFINE_string("out_dir", None, "out directory")
flags.DEFINE_string("expt_name", "default", "expt name")

flags.DEFINE_float("bc_scale", 1.0, "scale on random uniform bc")
flags.DEFINE_float("bc_weight", 100.0, "weight on bc loss")

flags.DEFINE_float(
    "relaxation_parameter",
    0.2,
    "Newton solver relaxation parameter",
)
flags.DEFINE_integer(
    "max_newton_steps",
    500,
    "Newton solver max steps",
)


# PDE
flags.DEFINE_string("pde", "poisson", "which PDE")
flags.DEFINE_integer(
    "ground_truth_resolution", 16, "mesh resolution for fenics ground truth"
)
flags.DEFINE_float(
    "boundary_resolution_factor",
    3.0,
    "ratio of resolution to points around boundary of shape",
)
flags.DEFINE_integer("fixed_num_pdes", None, "fixed number of pdes")
flags.DEFINE_integer(
    "n_eval", 16, "num eval PDEs",
)
flags.DEFINE_integer(
    "validation_points", 1024, "num points in domain for validation",
)
flags.DEFINE_boolean("vary_source", True, "")
flags.DEFINE_boolean("vary_bc", True, "")
flags.DEFINE_boolean("vary_geometry", True, "")
flags.DEFINE_boolean("vary_ic", True, "")
flags.DEFINE_string("domain_loss", None, "")

flags.DEFINE_float("tmin", 0.0, "PDE initial time")
flags.DEFINE_float("tmax", 1.0, "PDE final time")
flags.DEFINE_integer("num_tsteps", 101, "number of time steps for td_burgers")
flags.DEFINE_integer("sample_tsteps", 64, "number of time steps for td_burgers")
flags.DEFINE_boolean("sample_time_random", True, "random time sample for NN")
flags.DEFINE_float("max_reynolds", 100, "Reynolds number scale")
flags.DEFINE_string("burgers_pde", "default", "types of burgers equation")
flags.DEFINE_float("xmin", 0.0, "scale on random uniform bc")
flags.DEFINE_float("xmax", 1.0, "scale on random uniform bc")
flags.DEFINE_float("ymin", -1.0, "scale on random uniform bc")
flags.DEFINE_float("ymax", 1.0, "scale on random uniform bc")
flags.DEFINE_integer("max_holes", 12, "scale on random uniform bc")
flags.DEFINE_float("max_hole_size", 0.4, "scale on random uniform bc")


# Seed
flags.DEFINE_integer("seed", 0, "set random seed")

# Training
flags.DEFINE_integer(
    "outer_points", 256, "num support points on the boundary and in domain",
)
flags.DEFINE_integer(
    "inner_points", 256, "num support points on the boundary and in domain",
)
flags.DEFINE_integer("outer_steps", int(1e8), "num outer steps")
flags.DEFINE_integer("num_layers", 3, "num fcnn layers")
flags.DEFINE_integer("layer_size", 64, "fcnn layer size")
flags.DEFINE_boolean("siren", True, "use siren")
flags.DEFINE_float("grad_clip", 1e14, "max grad for clipping")
flags.DEFINE_float("siren_omega", 1.0, "siren_omega")
flags.DEFINE_float("siren_omega0", 3.0, "siren_omega0")
flags.DEFINE_boolean("log_scale", True, "io_scale")
flags.DEFINE_float("io_scale_lr_factor", 1e1, 'scale io lr by this factor')
flags.DEFINE_string("optimizer", "ranger", "adam or ranger, currently no adahess")

# Logging
flags.DEFINE_integer("viz_every", int(1e4), "plot every N steps")
flags.DEFINE_integer("val_every", int(1e2), "validate every N steps")
flags.DEFINE_integer("log_every", int(1e3), "tflog every N steps")
flags.DEFINE_integer("measure_grad_norm_every", int(1e3), "")

# Testing
flags.DEFINE_string("load_model_from_expt", None, "load pre-trained model")
