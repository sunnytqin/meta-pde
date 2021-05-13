def get_pde(pde_name):
    if pde_name == "poisson":
        from .poisson import poisson_def

        return poisson_def
    elif pde_name == "linear_stokes":
        from .linear_stokes import linear_stokes_def

        return linear_stokes_def
    elif pde_name == "nonlinear_stokes":
        from .nonlinear_stokes import nonlinear_stokes_def

        return nonlinear_stokes_def
    else:
        raise Exception("Invalid PDE {}".format(pde_name))
