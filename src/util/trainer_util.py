"""Util functions used by both Leap and Maml which dont have another home.

Consider moving these into subfolders.

Also try move more functions in here."""


import jax
import jax.numpy as np
import numpy as npo
import matplotlib.pyplot as plt

import fenics as fa

from . import jax_tools


def get_ground_truth_points(args, pde, pde_params_list, key):
    """Given a pdedef and list of pde parameters, sample points in the domain and
    evaluate the ground truth at those points."""
    fenics_functions = []
    coefs = []
    coords = []
    keys = jax.random.split(key, len(pde_params_list))
    for params, key in zip(pde_params_list, keys):
        ground_truth = pde.solve_fenics(params)
        ground_truth.set_allow_extrapolation(True)
        fn_coords = pde.sample_points_in_domain(key, args.validation_points, params)
        coefs.append(np.array([ground_truth(x) for x in fn_coords]))
        coords.append(fn_coords)
        ground_truth.set_allow_extrapolation(False)
        fenics_functions.append(ground_truth)
    return fenics_functions, np.stack(coefs, axis=0), np.stack(coords, axis=0)


def extract_coefs_by_dim(function_space, dofs, out, dim_idx=0):
    """Given the value of u (in R^d) at *each* dof coordinate, fill into 'out' the
    vector of dof values corresponding to interpolating u into function_space.

    We need this funtion because each dof corresponds to only one dim of u(x),
    so evaluating u(x) at every dof coordinate gives us d times as many values as
    needed.

    Depending on the function space, dof coordinates might not be repeated (e.g.
    at some x, there is only a dof for some but not all dimensions), and we need to
    use fenics API to extract the right components of u for that dof).
    """
    dofs = dofs.reshape(dofs.shape[0], -1)  # If DoFs is 1d, make it 2d
    if function_space.num_sub_spaces() > 0:
        for sidx in range(function_space.num_sub_spaces()):
            dim_idx = extract_coefs_by_dim(
                function_space.sub(sidx), dofs, out, dim_idx)
    else:
        if dim_idx >= dofs.shape[1]:
            raise Exception("Mismatch between Fenics fn space dim and dofs arr")
        out[function_space.dofmap().dofs()] = dofs[
            function_space.dofmap().dofs(), dim_idx
        ]
        dim_idx += 1
    return dim_idx


def compare_plots_with_ground_truth(
    model, pde, fenics_functions, params_stacked, get_final_model, meta_alg_def
):
    """Plot the solutions corresponding to ground truth, and corresponding to the
    meta-learned model after each of k gradient steps.
    """
    keys = jax.random.split(jax.random.PRNGKey(0), len(fenics_functions))
    N = len(fenics_functions)

    params_list = jax_tools.tree_unstack(params_stacked)

    assert N % 2 == 0

    for i in range(min([N, 8])):  # Don't plot more than 8 PDEs for legibility
        ground_truth = fenics_functions[i]

        plt.subplot(7, min([N, 8]), 1 + i)
        plt.axis("off")
        ground_truth.set_allow_extrapolation(False)
        pde.plot_solution(ground_truth, params_list[i])
        if i == 0:
            plt.ylabel("Truth")

        for j in range(6):
            plt.subplot(7, min([N, 8]), 1 + min([N, 8]) * (j + 1) + i)
            final_model = get_final_model(
                keys[i], model, params_list[i], j, meta_alg_def,
            )

            coords = ground_truth.function_space().tabulate_dof_coordinates()

            dofs = final_model(np.array(coords))

            out = npo.zeros(len(coords))

            extract_coefs_by_dim(ground_truth.function_space(), dofs, out)

            u_approx = fa.Function(ground_truth.function_space())

            fenics_shape = npo.array(u_approx.vector()).shape
            assert fenics_shape == out.shape, "Mismatched shapes {} and {}".format(
                fenics_shape, out.shape
            )

            u_approx.vector().set_local(out)

            pde.plot_solution(u_approx, params_list[i])
            plt.axis("off")
            if i == 0:
                plt.ylabel("Model: {} steps".format(j))
