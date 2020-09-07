import jax.numpy as np


def project_grads(pcgrad_magnitude, grad, *other_grads):
    for j in range(len(other_grads)):
        dot_prod = (grad * other_grads[j]).sum()
        proj_size = dot_prod / (other_grads[j] * other_grads[j] + 1e-14).sum()
        grad = grad - (pcgrad_magnitude * np.minimum(proj_size, 0.0) * other_grads[j])
    return grad
