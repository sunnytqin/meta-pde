import jax


def dict_flatten(x, prefix=None):
    """Given hierarchy of dicts, flatten them out into one dict.

    E.g.
    {'Layer1': {'Weight': W, 'Bias': b}}
    ->
    {'Layer1_Weight': W, 'Layer1_Bias': b}

    Useful for getting a list of params in a Flax model (e.g. for visualization)
    """
    if isinstance(x, dict):
        out = []
        for key in x.keys():
            out = out + dict_flatten(
                x[key], str(key) if prefix is None else prefix + "_" + str(key)
            )
        return out
    return [(prefix, x)]


def tree_unstack(x):
    """Unstacks a pytree x to a list of xi.

    If x is a valid input/output for a vmapped function fv,
    then each xi is a valid input for the non-vmapped original function f.

    The arrays at the leaf notes of the pytree x should each have the same
    leading dimension size. The list returned is of pytrees with the same
    structure as x but this leading dimension removed at each leaf array
    (i.e., each element in the list indexes one row of the leadning dim).

    Inputs:
        x: a jax pytree

    Outputs:
        list of xi
    """
    leaves = jax.tree_util.tree_leaves(x)
    dim = leaves[0].shape[0]
    for leaf in leaves:
        assert leaf.shape[0] == dim

    return [jax.tree_map(lambda x: x[i], x) for i in range(dim)]
