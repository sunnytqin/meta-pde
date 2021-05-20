import absl
from absl import app
from absl import flags

from jax.config import config; config.update("jax_log_compiles", 1)


FLAGS = flags.FLAGS

# Loggin
flags.DEFINE_string("out_dir", None, "out directory")
flags.DEFINE_string("expt_name", "default", "expt name")

flags.DEFINE_float("bc_scale", 1.0, "scale on random uniform bc")
flags.DEFINE_float("bc_weight", 100.0, "weight on bc loss")

flags.DEFINE_float(
    "relaxation_parameter",
    0.1,
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
flags.DEFINE_string("domain_loss", None, "")

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
flags.DEFINE_integer("layer_size", 256, "fcnn layer size")
flags.DEFINE_boolean("siren", True, "use siren")
flags.DEFINE_float("grad_clip", 1e14, "max grad for clipping")
flags.DEFINE_float("siren_omega", 1.0, "siren_omega")
flags.DEFINE_float("siren_omega0", 3.0, "siren_omega0")

flags.DEFINE_integer("viz_every", int(1e5), "plot every N steps")
flags.DEFINE_integer("val_every", int(1e2), "validate every N steps")
flags.DEFINE_integer("log_every", int(1e3), "tflog every N steps")
flags.DEFINE_integer("measure_grad_norm_every", int(1e3), "")

flags.DEFINE_string("optimizer", "ranger", "adam or ranger, currently no adahess")

flags.DEFINE_boolean("annealing", True, "annealing bc losses")
flags.DEFINE_boolean("annealing_l2", False, "anneal losses in l2 norm")

flags.DEFINE_float("annealing_alpha", 0.95, "annealing smoothing param")
