'''Simple, extensible Jax implementation of MAML.

The algorithm is from Model-Agnostic Meta Learning for Fast Adaptation of Deep
Networks, by Finn, Abbeel, and Levine, ICML 2017. https://arxiv.org/abs/1703.03400
'''

from functools import partial
from collections import namedtuple

import jax
import jax.numpy as np

import flax


# MamlDef contains algorithm-level parameters.
# Think of constructing MamlDef as akin to passing args to the __init__ of a Trainer
# class, except here we're using an (immutable) namedtuple instead of a class instance,
# and functions rather than class instance methods, to be more Jax-like
# and to be more obvious about the functions having no side effects.
MamlDef = namedtuple('MamlDef', [
    'make_inner_opt',       # fn: Flax model -> Flax optimizer
    'make_task_params',     # fn: PRNGKey -> params which define a task
    'inner_loss_fn',        # fn: key, model, task_params -> inner-loop loss
    'outer_loss_fn',        # fn: key, model, task_params -> outer-loop (meta) loss
    'inner_steps',          # int: num inner-loop optimization steps
    'n_batch_tasks'         # int: number of 'tasks' in a batch for a meta-train step
])


@partial(jax.jit, static_argnums=0)
def multitask_rollout(maml_def, key, base_model):
    '''Roll out meta learner across multiple tasks, collecting MAML gradients.

    Args:
        maml_def: MamlDef namedtuple
        key: terminal PRNGKey
        base_model: a Flax model

    Returns:
        grads: gradient, of same type/treedef as the Flax model
        losses: [n_tasks, n_steps] array of losses at each inner step of each task
    '''
    keys = jax.random.split(key, maml_def.n_batch_tasks)
    grads, losses, meta_losses = jax.vmap(single_task_grad_and_losses,
                             in_axes=(None, 0, None))(maml_def, keys, base_model)
    grads = jax.tree_util.tree_map(lambda g: g.mean(axis=0), grads)
    return grads, losses, meta_losses


@partial(jax.jit, static_argnums=0)
def single_task_grad_and_losses(maml_def, key, base_model):
    '''Roll out meta learner on one task, collecting MAML gradients,
       discarding final model.

    Args:
        maml_def: MamlDef namedtuple
        key: terminal PRNGKey
        base_model: a Flax model

    Returns:
        grads: gradient, of same type/treedef as the Flax model
        losses: [n_steps] array of losses at each inner step
        meta_loss: the test loss (on the given train task)
    '''
    _, meta_grad, losses, meta_loss = single_task_rollout(
        maml_def, key, base_model)
    return meta_grad, losses, meta_loss


def single_task_deploy(maml_def, model_to_loss_fn, base_model):
    '''Roll out meta learner on one task, WITHOUT collecting MAML gradients.

    Do not jit this function!
    I used a for loop here instead of scan.
    This could result in very long compile time if inner_steps is large.

    Args:
        maml_def: MamlDef namedtuple
        model_to_loss_fn: an externally supplied function which maps from a Flax
                    model to its loss. User must implement any randomness or
                    task-specific behavior within this function.
        base_model: a Flax model to use as initialization
                    (e.g. the model which has been meta-learned via Leap)

    Returns:
        final_model: trained model of same type as base_model
    '''
    opt = maml_def.make_inner_opt(base_model)
    for _ in range(maml_def.inner_steps):
        grad = jax.grad(model_to_loss_fn)(opt.target)
        opt = opt.apply_gradient(grad)

    final_model = opt.target
    return final_model


@partial(jax.jit, static_argnums=0)
def single_task_rollout(maml_def, key, base_model):
    '''Roll out meta learner on one task, collecting Leap gradients.

    Args:
        maml_def: MamlDef namedtuple
        key: PRNGKey
        base_model: a Flax model to use as initialization

    Returns:
        final_opt.target: trained model of same type as base_model
        meta_grad: accumulated MAML gradient
        losses: [n_steps + 1] array of losses at each step
        meta_loss: the test loss (on the given train task)
    '''
    params_key, loss_final_key, meta_loss_key, inner_key = jax.random.split(key, 4)
    inner_keys = jax.random.split(inner_key, maml_def.inner_steps)
    task_params = maml_def.make_task_params(params_key)

    def body_fn(carry, key):
        opt = carry
        opt, loss = maml_inner_step(maml_def, key, opt, task_params)
        return opt, loss

    def inner_loop(initial_model):
        inner_opt = maml_def.make_inner_opt(initial_model)
        final_opt, losses = jax.lax.scan(
            body_fn, inner_opt, inner_keys, length=maml_def.inner_steps)
        meta_loss = maml_def.outer_loss_fn(
            meta_loss_key, final_opt.target, task_params)
        return meta_loss, (final_opt, losses)

    (meta_loss, (final_opt, losses)), meta_grad = jax.value_and_grad(
        inner_loop, has_aux=True)(base_model)

    loss_final = maml_def.inner_loss_fn(loss_final_key, base_model, task_params)

    losses = np.concatenate((losses, np.array([loss_final])))

    return final_opt.target, meta_grad, losses, meta_loss


@partial(jax.jit, static_argnums=0)
def maml_inner_step(maml_def, key, opt, task_params):
    '''Inner step of MAML single-task rollout.'''

    # differentiate w.r.t arg1 because args to inner loss are (key, model, task_params)
    loss_and_grad_fn = jax.value_and_grad(maml_def.inner_loss_fn, argnums=1)

    loss, grad = loss_and_grad_fn(key, opt.target, task_params)
    new_opt = opt.apply_gradient(grad)
    return new_opt, loss


def run_sinusoid():
    '''Test the code on a simple sinusiod problem, a la MAML.'''

    # Sinusoid loss, with a given phase
    def loss_fn(key, model, task_params):
        phase = task_params
        x = jax.random.uniform(key, shape=(32, 2))
        y = np.sin(x[:,0].reshape([-1, 1]) + phase)
        yhat = model(x)
        return np.mean((y-yhat)**2)

    # Simple definition of an MLP with Swish activations
    @flax.nn.module
    def MLP(x):
        for _ in range(3):
            x = flax.nn.Dense(x, 64)
            x = flax.nn.swish(x)
        x = flax.nn.Dense(x, 1)
        return x

    # Create a base model and the meta-model optimizer
    _, initial_params = MLP.init_by_shape(
       jax.random.PRNGKey(0), [((1, 2), np.float32)])

    model = flax.nn.Model(MLP, initial_params)
    meta_opt = flax.optim.Adam(learning_rate=1e-3).create(model)

    # Create helper functions for the LeapTrainer
    # Fn to make parameters which define a task
    make_task_params = lambda key: jax.random.uniform(key, shape=(1, 1),
                                                      minval=0., maxval=2*np.pi)

    # Fn to make an inner optimizer from an initial model
    make_inner_opt = flax.optim.Momentum(learning_rate=0.1, beta=0.).create

    # Specify the MAML algorithm-level parameters
    maml_def = MamlDef(make_inner_opt, make_task_params,
                       inner_loss_fn=loss_fn, outer_loss_fn=loss_fn,
                       inner_steps=10, n_batch_tasks=32)

    # Run the meta-train loop
    key = jax.random.PRNGKey(1)
    for i in range(1000):
        key, subkey = jax.random.split(key)
        grad, losses, meta_losses = multitask_rollout(
            maml_def, subkey, meta_opt.target)
        print("meta-step {}, meta_loss {}, per-inner-step avg losses {}".format(
            i, np.mean(meta_losses), np.mean(losses, axis=0)))
        meta_opt = meta_opt.apply_gradient(grad)


if __name__ == '__main__':
    run_sinusoid()
