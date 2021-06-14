import jax
import jax.numpy as np
import numpy as npo
from jax import grad, jit, vmap

import flax

from flax import nn

from functools import partial
import pdb


def siren_init(omega):
    def init_fn(key, shape, dtype=np.float32, omega=omega):
        fan_in = shape[0]
        return jax.random.uniform(
            key,
            shape,
            dtype,
            -np.sqrt(6.0 / fan_in) / omega,
            np.sqrt(6.0 / fan_in) / omega,
        )

    return init_fn


def first_layer_siren_init(omega, omega0):
    def init_fn(key, shape, dtype=np.float32, omega=omega, omega0=omega0):
        fan_in = shape[0]
        return (omega0 / omega) * jax.random.uniform(
            key, shape, dtype, -1.0 / fan_in, 1.0 / fan_in
        )

    return init_fn


def vmap_laplace_operator(x, potential_fn, weighting_fn=lambda x: 1.0):
    to_vmap = lambda x, potential_fn=potential_fn, weighting_fn=weighting_fn: (
        laplace_operator(x, potential_fn, weighting_fn)
    )
    return vmap(to_vmap)(x)


def laplace_operator(x, potential_fn, weighting_fn=lambda x: 1.0):
    """
    Inputs:
        x: [2]
        potential_fn: fn which takes x and returns the potential phi

    returns:
        laplacian: laplacian of phi(x,y)
    """
    assert len(x.shape) == 1
    dtype = x.dtype
    # hess_fn = jax.jacfwd(jax.jacrev(lambda x: potential_fn(x).squeeze()))
    hess_fn = jax.jacfwd(
        lambda x2: jax.jacrev(lambda x1: potential_fn(x1).squeeze())(x2)
        * weighting_fn(x2)
    )
    hess = hess_fn(x)
    assert len(hess.shape) == 2
    return np.trace(hess)


def vmap_divergence(x, field_fn):
    to_vmap = lambda x, field_fn=field_fn: (divergence(x, field_fn))
    return vmap(to_vmap)(x)


def divergence(x, field_fn):
    """
    Inputs:
        x: [2]
        potential_fn: fn which takes x and returns the potential phi

    returns:
        divergence of u(x,y)
    """
    assert len(x.shape) == 1
    dtype = x.dtype
    # hess_fn = jax.jacfwd(jax.jacrev(lambda x: potential_fn(x).squeeze()))
    jac = jax.jacfwd(lambda x: field_fn(x).squeeze())(x)
    assert len(jac.shape) == 2
    return np.trace(jac)


def divergence_tensor(x, tensor_fn):
    assert len(x.shape) == 1
    jac = jax.jacfwd(lambda x: tensor_fn(x).squeeze())(x)
    assert len(jac.shape) == 3
    assert jac.shape[0] == jac.shape[1]
    assert jac.shape[1] == jac.shape[2]
    return np.trace(jac, axis1=1, axis2=2)


def vmap_divergence_tensor(x, tensor_fn):
    to_vmap = lambda x, tensor_fn=tensor_fn: (divergence_tensor(x, tensor_fn))
    return vmap(to_vmap)(x)


def fourier_features(x, n_features):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    assert len(x.shape) == 2
    x = x.reshape(x.shape[0], x.shape[1], 1)
    pows = np.arange(n_features).reshape(1, 1, -1)
    sins = sins = np.sin(2 ** pows * x) / 2 ** pows  # * 2 * np.pi)
    coss = coss = np.cos(2 ** pows * x) / 2 ** pows  # * 2 * np.pi)
    return np.concatenate([x, sins, coss], axis=-1).reshape(
        x.shape[0], -1
    )  # / n_features


def whiten(x, mean, std):
    if mean is not None:
        x = x - mean.reshape(1, -1)
    if std is not None:
        x = x / std.reshape(1, -1)
    return x


def dewhiten(y, mean, std):
    if std is not None:
        y = y * std.reshape(1, -1)
    if mean is not None:
        y = y + mean.reshape(1, -1)
    return y


def nf_apply(
    out_dim,
    x,
    sizes,
    dense_args=(),
    nonlinearity=nn.relu,
    mean_x=None,
    std_x=None,
    mean_y=None,
    std_y=None,
    n_fourier=None,
    omega=30.0,
    omega0=30.0,
):
    if nonlinearity == np.sin:
        kernel_init = siren_init(omega)
        first_init = first_layer_siren_init(omega, omega0)
    else:
        kernel_init = flax.nn.initializers.variance_scaling(
            1.0, "fan_in", "truncated_normal"
        )  # flax.nn.initializers.lecun_normal()
        first_init = kernel_init
    # x = whiten(x, mean_x, std_x)
    if n_fourier is not None:
        x = fourier_features(x, n_fourier)
    for i, size in enumerate(sizes):
        a = flax.nn.Dense(x, size, kernel_init=first_init if i == 0 else kernel_init)
        if nonlinearity == np.sin:
            # omega0 in siren, hacked so we can choose to only do it on first layer
            x = nonlinearity(a* omega)
        else:
            x = nonlinearity(a)
    out = flax.nn.Dense(x, out_dim, kernel_init=kernel_init,
                        bias_init=flax.nn.initializers.zeros)
    # out = flax.nn.Dense(out, out_dim, bias_init=flax.nn.initializers.zeros)
    return out  # dewhiten(out, mean_y, std_y)


class NeuralField2d(nn.Module):
    def apply(
        self, x, *args, **kwargs,
    ):
        return nf_apply(x.shape[-1], x, *args, **kwargs,)


NeuralField = NeuralField2d


class NeuralField1d(nn.Module):
    def apply(
        self, *args, **kwargs,
    ):
        return nf_apply(1, *args, **kwargs,).sum(axis=-1)


def make_nf_ndim(n_dims):
    class HigherDimNeuralField(nn.Module):
        def apply(
            self, *args, **kwargs,
        ):
            return nf_apply(n_dims, *args, **kwargs,)

    return HigherDimNeuralField


class DivFreeVelocityPressureField(nn.Module):
    def apply(
        self, x, *args, **kwargs,
    ):
        x_shape = x.shape
        x = x.reshape(-1, 2)

        base_field = NeuralField2d.shared(**kwargs)

        def phi_and_p_fn(x_):
            phi_p = base_field(x_)
            return np.sum(phi_p[:, 0]), phi_p[:, 1]

        gradphi, pressure = jax.grad(phi_and_p_fn, has_aux=True)(x)

        vel_x = gradphi[:, 1]
        vel_y = -gradphi[:, 0]

        return np.stack((vel_x, vel_y, pressure), axis=1).reshape((*x_shape[:-1], 3))

class DivFreeVelocityField(nn.Module):
    def apply(
            self, x, *args, **kwargs,
    ):
        x_shape = x.shape
        x = x.reshape(-1, 2)

        base_field = NeuralField1d.shared(**kwargs)

        def phi_fn(x_):
            phi_p = base_field(x_)
            return np.sum(phi_p)

        gradphi = jax.grad(phi_fn, has_aux=False)(x)

        vel_x = gradphi[:, 1]
        vel_y = -gradphi[:, 0]

        return np.stack((vel_x, vel_y), axis=1).reshape((*x_shape[:-1], 2))




NeuralPotential = NeuralField1d


@partial(jit, static_argnums=(1,))
def A_from_coefs(coefs, input_dim):
    A = np.zeros([coefs.shape[0], input_dim, input_dim])
    idxs = np.tril_indices(input_dim)
    ci = 0
    for i, j in zip(*idxs):
        A = jax.ops.index_add(A, jax.ops.index[:, i, j], coefs[:, ci])
        ci += 1
    A = A - np.transpose(A, (0, 2, 1))
    return A


@partial(jit, static_argnums=(1))
def make_svf(x, get_coefs_fn):
    out = np.zeros((x.shape[0], x.shape[1]))

    for i in range(x.shape[1]):
        for j in range(x.shape[1]):
            if x.shape[0] > 1:
                pdb.set_trace()

            def term_fn(x):
                return A_from_coefs(get_coefs_fn(x), x.shape[1])[:, i, j].sum()

            term = grad(term_fn)(x)[:, i]
            out = jax.ops.index_add(out, jax.ops.index[:, j], term)
    return out


class DivFreeField(nn.Module):
    def get_coefs(self, x, mean_x=None, std_x=None):
        x = whiten(x, mean_x, std_x)
        for layer in self.hidden_layers:
            a = layer(x)
            x = self.nonlinearity(a)
        coefs = self.output_layer(x)
        return coefs

    def apply(
        self,
        x,
        sizes,
        dense_args=(),
        nonlinearity=nn.relu,
        mean_x=None,
        std_x=None,
        mean_y=None,
        std_y=None,
    ):

        input_dim = x.shape[-1]

        self.hidden_layers = [flax.nn.Dense.shared(features=size) for size in sizes]
        self.output_layer = flax.nn.Dense.shared(
            features=input_dim * (input_dim - 1) // 2
        )
        self.nonlinearity = nonlinearity

        assert len(x.shape) == 2

        out = make_svf(x, partial(self.get_coefs, mean_x=mean_x, std_x=std_x))

        return dewhiten(out, mean_y, std_y)
