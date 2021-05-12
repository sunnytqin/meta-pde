from .nonlinear_stokes import nonlinear_stokes_def
from .poisson import poisson_def
from .linear_stokes import linear_stokes_def


def get_pde(pde_name):
    if pde_name == "poisson":
        return poisson_def
    elif pde_name == "linear_stokes":
        return linear_stokes_def
    elif pde_name == "nonlinear_stokes":
        return nonlinear_stokes_def
    else:
        raise Exception("Invalid PDE {}".format(pde_name))
