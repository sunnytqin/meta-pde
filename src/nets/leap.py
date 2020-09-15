'''Simple, extensible Jax implementation of Leap.

The algorithm is from Transferring Knowledge Across Learning Processes,
by Flennerhag, Moreno, Lawrence, and Damianou, ICLR 2019.
https://arxiv.org/abs/1812.01054

I used the author's code as reference: https://github.com/amzn/metalearn-leap
'''


from functools import partial
from collections import namedtuple

import jax
import jax.numpy as np

import flax


# LeapDef contains algorithm-level parameters.
# Think of constructing LeapDef as akin to passing args to the __init__ of a Trainer
# class, except here we're using an (immutable) namedtuple instead of a class instance,
# and functions rather than class instance methods, to be more Jax-like
# and to be more obvious about the functions having no side effects.
LeapDef = namedtuple('LeapDef', [
    'make_inner_opt',   # fn: Flax model -> Flax optimizer
    'make_task_params', # fn: PRNGKey -> params which define a task
    'loss_fn',          # fn: key, model, task_params -> loss
    'inner_steps',      # int: num inner-loop optimization steps
    'n_batch_tasks',    # int: number of 'tasks' in a batch for a meta-train step
    'norm',             # bool: whether to normalize Leap gradients
    'loss_in_distance', # bool: whether to use loss to compute distances on task manifold
    'stabilize'         # bool: whether to use a stabilizer for Leap grads
])  # See the paper and reference code to understand the last three args.


@partial(jax.jit, static_argnums=0)
def multitask_rollout(leap_def, key, base_model):
    '''Roll out meta learner across multiple tasks, collecting Leap gradients.

    Args:
        key: terminal PRNGKey
        base_model: a Flax model

    Returns:
        grads: gradient, of same type/treedef as the Flax model
        losses: [n_tasks, n_steps] array of losses at each inner step of each task
    '''
    keys = jax.random.split(key, leap_def.n_batch_tasks)
    grads, losses = jax.vmap(single_task_grad_and_losses,
                             in_axes=(None, 0, None))(leap_def, keys, base_model)
    grads = jax.tree_util.tree_map(lambda g: g.mean(axis=0), grads)
    return grads, losses


@partial(jax.jit, static_argnums=0)
def single_task_grad_and_losses(leap_def, key, base_model):
    '''Roll out meta learner on one task, collecting Leap gradients,
       discarding final model.

    Args:
        key: terminal PRNGKey
        base_model: a Flax model

    Returns:
        grads: gradient, of same type/treedef as the Flax model
        losses: [n_steps] array of losses at each inner step
    '''
    final_model, meta_grad, losses = single_task_rollout(leap_def, key, base_model)
    return meta_grad, losses


def single_task_deploy(leap_def, model_to_loss_fn, base_model):
    '''Roll out meta learner on one task, WITHOUT collecting Leap gradients.

    Do not jit this function!
    I used a for loop here instead of scan.
    This could result in very long compile time if inner_steps is large.

    Args:
        model_to_loss_fn: an externally supplied function which maps from a Flax
                    model to its loss. User must implement any randomness or
                    task-specific behavior within this function.
        base_model: a Flax model to use as initialization
                    (e.g. the model which has been meta-learned via Leap)

    Returns:
        final_model: trained model of same type as base_model
    '''
    opt = leap_def.make_inner_opt(base_model)
    for i in range(leap_def.inner_steps):
        grad = jax.grad(model_to_loss_fn)(opt.target)
        opt = opt.apply_gradient(grad)

    final_model = opt.target
    return final_model


@partial(jax.jit, static_argnums=0)
def single_task_rollout(leap_def, key, base_model):
    '''Roll out meta learner on one task, collecting Leap gradients.

    Args:
        key: PRNGKey
        base_model: a Flax model to use as initialization

    Returns:
        final_opt.target: trained model of same type as base_model
        meta_grad_accum: accumulated Leap gradient
        losses: [n_steps + 1] array of losses at each step
    '''
    params_key, loss0_key, inner_key = jax.random.split(key, 3)
    inner_keys = jax.random.split(inner_key, leap_def.inner_steps)
    task_params = leap_def.make_task_params(key)

    loss0 = leap_def.loss_fn(loss0_key, base_model, task_params)

    inner_opt = leap_def.make_inner_opt(base_model)

    meta_grad_accum = jax.tree_map(lambda x: x*0, inner_opt.target)

    def body_fn(carry, key):
        opt, meta_grad_accum = carry
        opt, meta_grad_accum, loss = leap_inner_step(
            leap_def, key, opt, task_params, meta_grad_accum)
        return (opt, meta_grad_accum), loss

    (final_opt, meta_grad_accum), losses = jax.lax.scan(
        body_fn, (inner_opt, meta_grad_accum), inner_keys, length=leap_def.inner_steps)

    losses = np.concatenate((np.array([loss0]), losses))

    return final_opt.target, meta_grad_accum, losses


@partial(jax.jit, static_argnums=0)
def leap_inner_step(leap_def, key, opt, task_params, meta_grad_accum):
    '''Inner step of Leap single-task rollout.'''

    k1, k2 = jax.random.split(key, 2)

    # differentiate w.r.t arg1 because args to inner loss are (key, model, task_params)
    loss_and_grad_fn = jax.value_and_grad(leap_def.loss_fn, argnums=1)

    loss, grad = loss_and_grad_fn(k1, opt.target, task_params)
    new_opt = opt.apply_gradient(grad)
    new_loss = leap_def.loss_fn(k2, new_opt.target, task_params)

    meta_grad_increment = get_meta_grad_increment(
        leap_def, new_opt.target, opt.target, new_loss, loss, grad
    )

    meta_grad_accum = jax.tree_util.tree_multimap(lambda x, y: x+y,
                                                  meta_grad_accum,
                                                  meta_grad_increment)
    return new_opt, meta_grad_accum, new_loss


@partial(jax.jit, static_argnums=0)
def get_meta_grad_increment(leap_def, new_model, model, new_loss, loss, grad):
    '''Get Leap meta-grad increment. See paper/author code for details.'''
    d_loss = new_loss - loss
    if leap_def.stabilize:
        d_loss = - np.abs(d_loss)

    if leap_def.norm:
        norm = compute_global_norm(leap_def, new_model, model, d_loss)
    else:
        norm = 1.

    meta_grad_increment = jax.tree_util.tree_multimap(
        lambda x, y: x-y, model, new_model)

    if leap_def.loss_in_distance:
        meta_grad_increment = jax.tree_util.tree_multimap(
            lambda x, y: x - d_loss * y, meta_grad_increment, grad)

    meta_grad_increment = jax.tree_util.tree_map(lambda x: x / norm,
                                                 meta_grad_increment)

    return meta_grad_increment


def compute_global_norm(leap_def, new_model, old_model, d_loss):
    '''Compute norm within task manifold. See paper for details.'''
    model_sq = jax.tree_util.tree_multimap(
        lambda x, y: np.sum((x-y)**2), new_model, old_model)
    sum_sq = jax.tree_util.tree_reduce(lambda x, y: x+y,
                                       model_sq)
    if leap_def.loss_in_distance:
        sum_sq = sum_sq + d_loss **2

    norm = np.sqrt(sum_sq)
    return norm


def run_sinusoid():
    '''Test the code on a simple sinusiod problem, a la MAML.'''

    # Sinusoid loss with different period
    def loss_fn(key, model, task_params):
        phase = task_params
        x = jax.random.uniform(key, shape=(32, 2))
        y = np.sin(x[:,0].reshape([-1, 1]) + phase)
        yhat = model(x)
        return np.mean((y-yhat)**2)

    # Simple definition of an MLP with Swish activations
    @flax.nn.module
    def MLP(x):
        for i in range(3):
            x = flax.nn.Dense(x, 64)
            x = flax.nn.swish(x)
        x = flax.nn.Dense(x, 1)
        return x

    # Create a base model and the meta-model optimizer
    _, initial_params = MLP.init_by_shape(
       jax.random.PRNGKey(0), [((1, 2), np.float32)])

    model = flax.nn.Model(MLP, initial_params)
    meta_opt = flax.optim.Adam(learning_rate=1e-3).create(model)

    # Create helper functions needed for the LeapDef

    # Fn to make parameters which define a task
    make_task_params = lambda key: jax.random.uniform(key, shape=(1, 1),
                                                      minval=0., maxval=2*np.pi)

    # Fn to make an inner optimizer from an initial model
    make_inner_opt = flax.optim.Momentum(learning_rate=0.1, beta=0.).create

    # Create the LeapDef
    leap_def = LeapDef(make_inner_opt, make_task_params, loss_fn,
                       inner_steps=10, n_batch_tasks=32,
                       norm=True, loss_in_distance=True, stabilize=True)

    # Run the meta-train loop
    key = jax.random.PRNGKey(1)
    for i in range(1000):
        key, subkey = jax.random.split(key)
        grad, losses = multitask_rollout(leap_def, subkey, meta_opt.target)
        print("meta-step {}, per-inner-step avg losses {}".format(i,
                                                                  np.mean(losses,
                                                                          axis=0)))
        meta_opt = meta_opt.apply_gradient(grad)



if __name__ == '__main__':
    run_sinusoid()
