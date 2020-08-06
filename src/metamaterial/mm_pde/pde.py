"""Base class for PDEs"""
import fenics as fa
from .solver import Solver


class PDE(object):
    """Base class for PDEs.

    This defines init structure and the solve() routine.

    Subclasses should implement _build_mesh, _build_function_space,
    and _energy_density.

    Assumes the PDE is in a form where solving the pde is equivalent to
    minimizing the integral of energy_density(u) + external_work_fn(u),
    where external_work_fn and boundary conditions are supplied as arguments
    to solve().

    To define a PDE with a custom mesh, provide mesh to the constructor. Useful
    when e.g. defining a metamaterial / nonlinear elasticity PDE, and want to
    override default mesh construction so that there is no pore (e.g. to
    create a simple Fenics surrogate to use as baseline or compose with NN)
    """

    def __init__(self, args, mesh=None):
        # TODO (alex): make args non-optional
        self.args = args
        if mesh is None:
            self._build_mesh()
        else:
            self.mesh = mesh
        self._build_function_space()
        self._create_boundary_measure()

    def _create_boundary_measure(self):
        exterior_domain = fa.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        exterior_domain.set_all(0)
        self.exterior.mark(exterior_domain, 1)
        self.boundary_ds = fa.Measure("ds")(subdomain_data=exterior_domain)(1)

    def _build_mesh(self):
        raise NotImplementedError()

    def _build_function_space(self):
        raise NotImplementedError()

    def _energy_density(self, u):
        raise NotImplementedError()

    def energy(self, u):
        return fa.assemble(self._energy_density(u) * fa.dx)

    def solve_problem(
        self,
        args,
        boundary_fn=None,
        boundary_fn_dic=None,
        external_work_fn=None,
        external_work_fn_dic=None,
        initial_guess=None,
    ):
        u = fa.Function(self.V)
        delta_u = fa.Function(self.V)
        du = fa.TrialFunction(self.V)
        v = fa.TestFunction(self.V)

        if initial_guess is not None:
            # We shouldn't need this assert, but Fenics lets you pass iterables
            # of the wrong size, and just sets the elems that match, which
            # blows up solve if you pass an initial guess vector in the wrong
            # basis (as it sets the wrong elements)
            assert len(initial_guess) == len(u.vector())
            u.vector().set_local(initial_guess)

        E = self._energy_density(u) * fa.dx

        # If boundary functions are defined using one global function
        if boundary_fn is not None:
            boundary_bc = fa.DirichletBC(self.V, boundary_fn, self.exterior)
            bcs = [boundary_bc]
        else:
            bcs = []

        # If boundary functions are defined separately for four edges
        if boundary_fn_dic is not None:
            for key in boundary_fn_dic:
                boundary_bc = fa.DirichletBC(
                    self.V, boundary_fn_dic[key], self.exteriors_dic[key]
                )
                bcs = bcs + [boundary_bc] if bcs else [boundary_bc]

        if external_work_fn is not None:
            E = E - external_work_fn(u) * self.boundary_ds

        if external_work_fn_dic is not None:
            for key in external_work_fn_dic:
                E = E - external_work_fn_dic[key](u) * self.ds(
                    self.boundaries_id_dic[key]
                )

        dE = fa.derivative(E, u, v)
        jacE = fa.derivative(dE, u, du)

        snes_args = {
            "method": args.snes_method,
            "linear_solver": args.linear_solver,
            "maximum_iterations": args.max_snes_iter,
        }
        newton_args = {
            "relaxation_parameter": args.relaxation_parameter,
            "linear_solver": args.linear_solver,
            "maximum_iterations": args.max_newton_iter,
        }
        solver_args = {
            "nonlinear_solver": args.nonlinear_solver,
            "snes_solver": snes_args,
            "newton_solver": newton_args,
        }

        fa.parameters["form_compiler"]["cpp_optimize"] = True
        ffc_options = {
            "optimize": True,
            "eliminate_zeros": True,
            "precompute_basis_const": True,
            "precompute_ip_const": True,
        }

        fa.solve(
            dE == 0,
            u,
            bcs,
            J=jacE,
            solver_parameters=solver_args,
            form_compiler_parameters=ffc_options,
        )

        return u
