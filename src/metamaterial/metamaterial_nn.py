import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

from field import NeuralField

from functools import partial
import flax
from flax import nn
from timer import Timer

from .metamaterial_fenics import solve_fenics, make_fenics
from .metamaterial_common import *
import fenics as fa

import matplotlib.pyplot as plt
import pdb

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--outer_lr", type=float, default=3e-5, help="outer learning rate")
parser.add_argument(
    "--boundary_points",
    type=int,
    default=512,
    help="num points on the boundary for inner loss",
)
parser.add_argument(
    "--domain_points",
    type=int,
    default=512,
    help="num points inside the domain for inner loss",
)
parser.add_argument("--outer_steps", type=int, default=100, help="num outer steps")
parser.add_argument("--num_layers", type=int, default=3, help="num fcnn layers")
parser.add_argument("--n_fourier", type=int, default=None, help="num fourier features")
parser.add_argument("--layer_size", type=int, default=512, help="fcnn layer size")
parser.add_argument("--vary_source", type=int, default=0, help="1 for true")
parser.add_argument("--vary_bc", type=int, default=0, help="1 for true")
parser.add_argument("--vary_geometry", type=int, default=0, help="1=true.")
parser.add_argument("--bc_scale", type=float, default=0.1, help="1 for true")
parser.add_argument("--interior_weight", type=float, default=1e-3,
                    help="weight on interior boundary loss")



args = parser.parse_args()


# Deformation gradient
def defgrad(u):
    """
    Inputs:
        u: fn mapping [2]  -> [2]
    Outputs:
        F: fn mapping [2] -> [2, 2]
    """

    def _dg(x):
        x = x.reshape(2)
        dudx = jax.jacfwd(lambda x: u(x).reshape(2))(x)
        # pdb.set_trace()
        assert dudx.shape[0] == 2 and dudx.shape[1] == 2
        return dudx + np.eye(2)

    return _dg


def neo_hookean_energy(u, source_params, x):
    """
    Inputs:
        u: fn mapping [n, 2] or [2] -> [n, 2] or [2]
        young_mod: scalar
        poisson_ratio: scalar

    Outputs:
        energy: [n, 1] or [1]
    """
    young_mod, poisson_ratio = source_params
    if poisson_ratio >= 0.5:
        raise ValueError(
            "Poisson's ratio must be below isotropic upper limit 0.5. Found {}".format(
                poisson_ratio
            )
        )
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))

    x = x.reshape(2)
    F = defgrad(u)(x)
    J = np.linalg.det(F)
    rcg = np.matmul(F.transpose(), F)  # right cauchy green
    Jinv = 1.0 / (J + 1e-14)
    I1 = np.trace(rcg)  # first invariant

    energy = (shear_mod / 2) * (Jinv * I1 - 2) + (bulk_mod / 2) * (J - 1) ** 2

    return energy


def inv2d(A):
    assert len(A.shape) == 2 and A.shape[0] == 2 and A.shape[1] == 2
    return 1./(A[0,0]*A[1,1] - A[0,1]*A[1,0]) * np.array(
        [[A[1,1], A[1,0]],
         [A[0,1] , A[0,0]]]
    )


def first_pk_stress(u, source_params, x):

    young_mod, poisson_ratio = source_params
    if poisson_ratio >= 0.5:
        raise ValueError(
            "Poisson's ratio must be below isotropic upper limit 0.5. Found {}".format(
                poisson_ratio
            )
        )
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))

    x = x.reshape(2)
    F = defgrad(u)(x)
    J = np.linalg.det(F)
    rcg = np.matmul(F.transpose(), F)  # right cauchy green
    Jinv = 1.0 / (J + 1e-14)
    I1 = np.trace(rcg)  # first invariant

    FinvT = inv2d(F).transpose()
    first_pk_stress = (
        Jinv * shear_mod * (F - (1 / 2) * I1 * FinvT)
        + J * bulk_mod * (J - 1) * FinvT
    )
    return first_pk_stress


def second_pk_stress(u, source_params, x):

    young_mod, poisson_ratio = source_params
    if poisson_ratio >= 0.5:
        raise ValueError(
            "Poisson's ratio must be below isotropic upper limit 0.5. Found {}".format(
                poisson_ratio
            )
        )
    shear_mod = young_mod / (2 * (1 + poisson_ratio))
    bulk_mod = young_mod / (3 * (1 - 2 * poisson_ratio))

    x = x.reshape(2)
    F = defgrad(u)(x)
    J = np.linalg.det(F)
    rcg = np.matmul(F.transpose(), F)  # right cauchy green
    Jinv = 1.0 / (J + 1e-14)
    I1 = np.trace(rcg)  # first invariant

    Finv = inv2d(F)
    first_pk_stress = (
        Jinv * shear_mod * (F - (1 / 2) * I1 * Finv.transpose())
        + J * bulk_mod * (J - 1) * Finv.transpose()
    )
    return np.matmul(Finv, first_pk_stress)


def vmap_nhe(x, u, source_params):
    return vmap(partial(neo_hookean_energy, u, source_params))(x).reshape(-1, 1)


def interior_bc(geo_params, source_params, u, x):
    """This loss returns <dEnergy/dx, N> where N is the normal to the boundary.
    It is a Nietsche boundary condition.
    """
    c1, c2 = geo_params
    assert x.shape[0] == 2 and len(x.shape) == 1

    theta = np.arctan2(x[1], x[0])

    def get_xy(theta):
        rval = r(theta, c1, c2)
        return np.array([rval*np.cos(theta), rval*np.sin(theta)])

    tangent = jax.jacfwd(get_xy)(theta)

    tangent = tangent.reshape(2)

    # <[a, b], [-b, a]> = ab - ba = 0
    normal = np.array([-tangent[1], tangent[0]])
    normal = (normal / np.linalg.norm(normal)).reshape(2, 1)

    # Non-stationary dispacements, i.e. du_dx = 0
    pk = first_pk_stress(u, source_params, x)  # dForce / dx
    assert len(pk.shape) == 2 and pk.shape[0] == 2 and pk.shape[1] == 2

    pk_dot_n = np.matmul(pk, normal)

    return np.sum(pk_dot_n**2)


def interior_bc_loss(points_on_interior_boundary, u, geo_params, source_params):
    # pdb.set_trace()
    lossval = vmap(partial(interior_bc, geo_params, source_params, u))(
        points_on_interior_boundary
    ).mean()
    return lossval


def boundary_loss_fn(points_on_boundary, field_fn, bc_params):
    err_on_boundary = vmap_boundary_conditions(points_on_boundary, bc_params).reshape(
        -1, 2
    ) - field_fn(points_on_boundary).reshape(-1, 2)
    loss_on_boundary = (err_on_boundary ** 2).mean()
    return loss_on_boundary


def domain_loss_fn(points_in_domain, field_fn, source_params):
    err_in_domain = vmap_nhe(points_in_domain, field_fn, source_params).reshape(-1, 1)
    loss_in_domain = (err_in_domain ** 2).mean()
    return loss_in_domain

'''
@partial(jax.jit, static_argnums=(3, 4,))
def loss_fn(points_on_boundary, points_in_domain, field_fn, source_params, bc_params):
    domain_loss = domain_loss_fn(points_in_domain, field_fn, source_params)
    boundary_loss = boundary_loss_fn(points_on_boundary, field_fn, bc_params)
    return domain_loss + boundary_loss, (domain_loss, boundary_loss)
'''

@partial(jax.jit, static_argnums=(4, 5, 6))
def train_step(
    points_in_domain, points_on_boundary, points_on_interior_boundary,
    optimizer, source_params, bc_params, interior_weight
):
    # pdb.set_trace()

    boundary_loss, boundary_grad = jax.value_and_grad(
        lambda model: boundary_loss_fn(points_on_boundary, model, bc_params)
    )(optimizer.target)

    bgrad_flat, treedef = jax.tree_flatten(boundary_grad)

    boundary_grad_norm = sum([np.linalg.norm(g) for g in bgrad_flat])

    domain_loss, domain_grad = jax.value_and_grad(
        lambda model: domain_loss_fn(points_in_domain, model, source_params)
    )(optimizer.target)

    dgrad_flat, treedef = jax.tree_flatten(domain_grad)

    domain_grad_norm = sum([np.linalg.norm(g) for g in dgrad_flat])

    interior_boundary_loss, interior_boundary_grad = jax.value_and_grad(
        lambda model: interior_weight * interior_bc_loss(
            points_on_interior_boundary, model, geo_params, source_params)
    )(optimizer.target)

    igflat, treedef = jax.tree_flatten(interior_boundary_grad)

    interior_boundary_grad_norm = sum([np.linalg.norm(g) for g in igflat])

    total_grad = jax.tree_unflatten(
        treedef, [g1 + g2 + g3 for g1, g2, g3 in zip(bgrad_flat, dgrad_flat, igflat)]
    )

    print("domain loss {}, domain grad {}".format(domain_loss, domain_grad_norm))
    print(
        "boundary loss {}, boundary grad {}".format(boundary_loss, boundary_grad_norm)
    )
    print(
        "interior boundary loss {}, interior boundary grad {}".format(
            interior_boundary_loss, interior_boundary_grad_norm)
    )
    optimizer = optimizer.apply_gradient(total_grad)
    grad_flat, _ = jax.tree_flatten(total_grad.params)
    total_grad_norm = sum([np.linalg.norm(g) for g in grad_flat])

    return (
        optimizer,
        domain_loss + boundary_loss + interior_boundary_loss,
        sum([np.linalg.norm(g) for g in grad_flat]),
    )


Field = NeuralField.partial(
    sizes=[args.layer_size for _ in range(args.num_layers)],
    dense_args=(),
    nonlinearity=lambda x: nn.swish(x) / 1.1,
    n_fourier=args.n_fourier
)

key, subkey = jax.random.split(jax.random.PRNGKey(0))

_, init_params = Field.init_by_shape(subkey, [((1, 2), np.float32)])
model = flax.nn.Model(Field, init_params)

optimizer = flax.optim.Adam(learning_rate=args.outer_lr).create(model)

with Timer() as t:
    key, subkey = jax.random.split(key)

    source_params, bc_params, geo_params = sample_params(subkey, args)
print("made params in {}s".format(t.interval))

with Timer() as t:
    ground_truth = solve_fenics(source_params, bc_params, geo_params)
    ground_truth.set_allow_extrapolation(True)
print("made ground truth in {}s".format(t.interval))

plt.figure()
plt.subplot(3, 1, 1)
opt_fenics = make_fenics(optimizer.target, geo_params)
fa.plot(opt_fenics, mode="displacement", title="neural")
plt.subplot(3, 1, 2)
fa.plot(ground_truth, mode="displacement", title="truth")
plt.subplot(3, 1, 3)
diff = fa.project(opt_fenics - ground_truth, ground_truth.function_space())
fa.plot(diff,
    mode="displacement",
    title="difference",
)
plt.savefig('mm_init.png')

k1, k2, k3 = jax.random.split(key, 3)
points_in_domain = sample_points_in_domain(k1, args.domain_points, geo_params)
points_on_boundary = sample_points_on_boundary(k2, args.boundary_points, geo_params)
points_on_interior_boundary = sample_points_on_interior_boundary(k3,
                                                           args.boundary_points,
                                                           geo_params)
plt.figure()
plt.scatter(points_in_domain[:, 0], points_in_domain[:, 1], label='domain')
plt.scatter(points_on_boundary[:, 0], points_on_boundary[:, 1], label='boundary')
plt.scatter(points_on_interior_boundary[:, 0], points_on_interior_boundary[:, 1], label='inner_boundary')
plt.legend()
plt.savefig('mm_sampling.png')

for step in range(args.outer_steps):
    key, sk1, sk2, sk3 = jax.random.split(key, 4)

    points_in_domain = sample_points_in_domain(sk1, args.domain_points, geo_params)
    points_on_boundary = sample_points_on_boundary(
        sk2, args.boundary_points, geo_params
    )
    points_on_interior_boundary = sample_points_on_interior_boundary(sk3,
                                                               args.boundary_points,
                                                               geo_params)
    optimizer, loss, grad_norm = train_step(
        points_in_domain, points_on_boundary, points_on_interior_boundary,
        optimizer, source_params, bc_params, args.interior_weight
    )
    preds = optimizer.target(points_in_domain)

    try:
        true = np.array([ground_truth(point) for point in points_in_domain]).reshape(
            preds.shape
        )
        supervised_rmse = ((preds - true)**2).mean()
        # pdb.set_trace()
    except Exception as e:
        pdb.set_trace()
    print(
        "step {}, loss {}, gnorm {}, rmse {}".format(
            step, float(loss), grad_norm, supervised_rmse
        )  # , supervised_rmse)
    )
    print("\n\n")


plt.figure()
plt.subplot(3, 1, 1)
fa.plot(make_fenics(optimizer.target, geo_params), mode="displacement", title="neural")
plt.subplot(3, 1, 2)
fa.plot(ground_truth, mode="displacement", title="truth")
plt.subplot(3, 1, 3)
fa.plot(
    make_fenics(
        lambda xs: optimizer.target(xs).reshape(-1, 2)
        - np.array([ground_truth(*x) for x in xs]).reshape(-1, 2),
        geo_params,
    ),
    mode="displacement",
    title="difference",
)
plt.savefig('mm_final.png')
