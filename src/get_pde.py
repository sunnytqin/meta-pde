from absl import flags
from .poisson import poisson_def
from .stokes import stokes_def
from .burgers import burgers_def



FLAGS = flags.FLAGS


def get_pde(pde_name):
    if pde_name == "poisson":
        FLAGS.domain_loss = "domain_loss"
        return poisson_def

    elif pde_name == "burgers":
        FLAGS.domain_loss = "domain_loss"
        return burgers_def

    elif pde_name == "linear_stokes":
        FLAGS.stokes_nonlinear = False
        FLAGS.domain_loss = "loss_stress"
        return stokes_def

    elif pde_name == "nonlinear_stokes":
        FLAGS.stokes_nonlinear = True
        FLAGS.domain_loss = "loss_stress"
        return stokes_def

    else:
        raise Exception("Invalid PDE {}".format(pde_name))
