from absl import flags
from .poisson import poisson_def
from .burgers import td_burgers_def
from .elasticity import hyper_elasticity_def


FLAGS = flags.FLAGS


def get_pde(pde_name):
    if pde_name == "poisson":
        FLAGS.domain_loss = "domain_loss"
        return poisson_def

    elif pde_name == "td_burgers":
        FLAGS.domain_loss = ["loss_domain", "loss_initial"]
        return td_burgers_def

    elif pde_name == "hyper_elasticity":
        FLAGS.domain_loss = "loss_domain"
        return hyper_elasticity_def

    else:
        raise Exception("Invalid PDE {}".format(pde_name))
