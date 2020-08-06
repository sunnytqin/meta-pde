"""Metamaterial PDE definition"""

import math
import numpy as np
import mshr
import fenics as fa
from .pde import PDE
from .strain import NeoHookeanEnergy


class Metamaterial(PDE):
    def _build_mesh(self):
        """Create mesh with pores defined by c1, c2 a la Overvelde&Bertoldi"""
        args = self.args
        (
            L0,
            porosity,
            c1,
            c2,
            resolution,
            n_cells,
            min_feature_size,
            pore_radial_resolution,
        ) = (
            args.L0,
            args.porosity,
            args.c1,
            args.c2,
            args.metamaterial_mesh_size,
            args.n_cells,
            args.min_feature_size,
            args.pore_radial_resolution,
        )

        r0 = L0 * math.sqrt(2 * porosity) / math.sqrt(math.pi * (2 + c1 ** 2 + c2 ** 2))

        def coords_fn(theta):
            return r0 * (1 + c1 * fa.cos(4 * theta) + c2 * fa.cos(8 * theta))

        base_pore_points, radii, thetas = build_base_pore(
            coords_fn, pore_radial_resolution
        )

        verify_params(base_pore_points, radii, L0, min_feature_size)

        material_domain = None
        pore_domain = None

        center_offset = L0 * n_cells * 0.5  # Make it centered on the origin

        for i in range(n_cells):
            for j in range(n_cells):
                pore = build_pore_polygon(
                    base_pore_points,
                    offset=(
                        L0 * (i + 0.5) - center_offset,
                        L0 * (j + 0.5) - center_offset,
                    ),
                )

                pore_domain = pore if not pore_domain else pore + pore_domain

                cell = mshr.Rectangle(
                    fa.Point(L0 * i - center_offset, L0 * j - center_offset),
                    fa.Point(
                        L0 * (i + 1) - center_offset, L0 * (j + 1) - center_offset
                    ),
                )
                material_in_cell = cell - pore
                material_domain = (
                    material_in_cell
                    if not material_domain
                    else material_in_cell + material_domain
                )

        mesh = mshr.generate_mesh(material_domain, resolution * n_cells)

        self.mesh = mesh

    def _build_function_space(self):
        """Create 2d VectorFunctionSpace and an exterior domain"""
        L0 = self.args.L0
        n_cells = self.args.n_cells

        class Exterior(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and (
                    fa.near(x[1], (L0 * n_cells / 2.0))
                    or fa.near(x[0], (L0 * n_cells / 2.0))
                    or fa.near(x[0], -(L0 * n_cells / 2.0))
                    or fa.near(x[1], -(L0 * n_cells / 2.0))
                )

        class Left(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0], -(L0 * n_cells / 2.0))

        class Right(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[0], (L0 * n_cells / 2.0))

        class Bottom(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1], -(L0 * n_cells / 2.0))

        class Top(fa.SubDomain):
            def inside(self, x, on_boundary):
                return on_boundary and fa.near(x[1], (L0 * n_cells / 2.0))

        self.exteriors_dic = {
            "left": Left(),
            "right": Right(),
            "bottom": Bottom(),
            "top": Top(),
        }
        self.exterior = Exterior()
        self.V = fa.VectorFunctionSpace(self.mesh, "P", 2)

        self.sub_domains = fa.MeshFunction(
            "size_t", self.mesh, self.mesh.topology().dim() - 1
        )
        self.sub_domains.set_all(0)

        self.boundaries_id_dic = {"left": 1, "right": 2, "bottom": 3, "top": 4}
        self.left = Left()
        self.left.mark(self.sub_domains, 1)
        self.right = Right()
        self.right.mark(self.sub_domains, 2)
        self.bottom = Bottom()
        self.bottom.mark(self.sub_domains, 3)
        self.top = Top()
        self.top.mark(self.sub_domains, 4)

        self.normal = fa.FacetNormal(self.mesh)

        self.ds = fa.Measure("ds")(subdomain_data=self.sub_domains)

    def _energy_density(self, u, return_stress=False):
        """Energy density is NeoHookean strain energy. See strain.py for def."""
        return NeoHookeanEnergy(
            u, self.args.young_modulus, self.args.poisson_ratio, return_stress
        )


""" Helper functions """


def build_base_pore(coords_fn, n_points):
    thetas = [float(i) * 2 * math.pi / n_points for i in range(n_points)]
    radii = [coords_fn(float(i) * 2 * math.pi / n_points) for i in range(n_points)]
    points = [
        (rtheta * np.cos(theta), rtheta * np.sin(theta))
        for rtheta, theta in zip(radii, thetas)
    ]
    return np.array(points), np.array(radii), np.array(thetas)


def build_pore_polygon(base_pore_points, offset):
    points = [fa.Point(p[0] + offset[0], p[1] + offset[1]) for p in base_pore_points]
    pore = mshr.Polygon(points)
    return pore


def verify_params(pore_points, radii, L0, min_feature_size):
    """Verify that params correspond to a geometrically valid structure"""
    # check Constraint A
    tmin = L0 - 2 * pore_points[:, 1].max()
    if tmin / L0 <= min_feature_size:
        raise ValueError(
            "Minimum material thickness violated. Params do not "
            "satisfy Constraint A from Overvelde & Bertoldi"
        )

    # check Constraint B
    # Overvelde & Bertoldi check that min radius > 0.
    # we check it is > min_feature_size > 2.0, so min_feature_size can be used
    # to ensure the material can be fabricated
    if radii.min() <= min_feature_size / 2.0:
        raise ValueError(
            "Minimum pore thickness violated. Params do not "
            "satisfy (our stricter version of) Constraint B "
            "from Overvelde & Bertoldi"
        )
