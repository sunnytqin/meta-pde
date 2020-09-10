import jax
import jax.numpy as np
from jax import grad, jit, vmap

from .field import fourier_features, siren_init, first_layer_siren_init

from functools import partial

import flax

from flax import nn

from jax.experimental import optimizers

import pdb


class GradientConditionedModel(nn.Module):
    def make_params(self, make_params_args):
        pass

    def base_forward(self, x, params):
        pass

    def apply(
        self,
        x,
        inner_loss_kwargs,
        inner_steps,
        inner_loss,
        base_args,
        inner_lr,
        train_inner_lrs,
        n_fourier=None,
    ):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        def param_loss(params):
            return inner_loss(
                partial(self.base_forward, params=params, n_fourier=n_fourier),
                **inner_loss_kwargs
            )

        scale_loss = lambda scale: lambda params: scale * param_loss(params)

        params = self.make_params(n_fourier=n_fourier, **base_args)

        if inner_steps > 0:
            opt_init, opt_update, get_params = optimizers.sgd(step_size=inner_lr)
            opt_state = opt_init(params)

            if train_inner_lrs:
                lrs = self.param("lrs", (inner_steps,), flax.nn.initializers.ones)
            else:
                lrs = np.array([1.0 for _ in range(inner_steps)])

            @jit
            def inner_step(opt_state, step_and_lr):
                step, lr = step_and_lr
                params = get_params(opt_state)
                lr_scaled_loss = scale_loss(lr)
                scaled_loss, gradient = jax.value_and_grad(lr_scaled_loss)(params)
                opt_state = opt_update(step, gradient, opt_state)
                return opt_state, scaled_loss

            opt_state, losses = jax.lax.scan(
                inner_step,
                opt_state,
                np.stack([np.arange(len(lrs)), lrs], axis=1),
                length=inner_steps,
            )

            params = get_params(opt_state)

        return self.base_forward(x, params, n_fourier)


class GradientConditionedField(GradientConditionedModel):
    def make_params(
        self,
        sizes,
        input_dim,
        output_dim,
        nonlinearity=nn.swish,
        kernel_init=flax.nn.initializers.variance_scaling(
            2.0, "fan_in", "truncated_normal"
        ),
        bias_init=flax.nn.initializers.zeros,
        n_fourier=None,
    ):
        self.nonlinearity = nonlinearity
        if self.nonlinearity == np.sin:
            kernel_init = siren_init
            first_init = first_layer_siren_init
        else:
            first_init = kernel_init
        self.input_dim = input_dim
        self.output_dim = output_dim
        if n_fourier is not None:
            self.sizes = (
                [n_fourier * 2 * input_dim]
                + sizes
                + [2 * input_dim * output_dim + output_dim]
            )
        else:
            self.sizes = (
                [input_dim] + sizes + [output_dim]
            )  # [2*input_dim*output_dim + output_dim]
        del sizes
        weights = []
        biases = []
        for i in range(len(self.sizes) - 1):
            weights.append(
                self.param(
                    "W" + str(i), (self.sizes[i], self.sizes[i + 1]),
                    first_init if i==0 else kernel_init
                )
            )
            biases.append(self.param("b" + str(i), (1, self.sizes[i + 1]), bias_init))
        return {"weights": weights, "biases": biases}

    def base_forward(self, x, params, n_fourier):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
            squeeze = True
        else:
            squeeze = False
        inputs = x
        if n_fourier is not None:
            x = fourier_features(x, n_fourier)
        a = x
        for W, b in zip(params["weights"], params["biases"]):
            x = np.dot(a, W) + b
            if self.nonlinearity == np.sin:
                a = a * 30.
            a = self.nonlinearity(x)
        """
        bout = x[:, :self.output_dim].reshape(-1, self.output_dim, 1)
        quads = x[:, self.output_dim:
                  self.output_dim+self.output_dim*self.input_dim].reshape(
                      x.shape[0], self.output_dim, self.input_dim)
        lins = x[:, self.output_dim+self.output_dim*self.input_dim:].reshape(
                      x.shape[0], self.output_dim, self.input_dim)
        inputs = inputs.reshape(inputs.shape[0], 1, -1)
        out = np.sum(inputs**2 * quads + inputs * lins + bout, axis=-1)
        """
        if squeeze:
            assert x.shape[0] == 1
            x = x.squeeze(0)
        return x
