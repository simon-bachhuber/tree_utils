from functools import partial, reduce
from typing import Generic, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox import tree_at

T = TypeVar("T")


class PyTree(Generic[T]):
    pass

    def __iter__(self):
        pass

    def __next__(self):
        pass


def add_batch_dim(values: PyTree) -> PyTree[jnp.ndarray]:
    return jax.tree_map(lambda x: jnp.expand_dims(x, axis=0), values)


def _flatten(x: jnp.ndarray, num_batch_dims: int) -> jnp.ndarray:
    if x.ndim < num_batch_dims:
        return x
    return jnp.reshape(x, list(x.shape[:num_batch_dims]) + [-1])


def tree_close(a: PyTree, b: PyTree, rtol=1e-05, atol=1e-08) -> bool:
    return all(
        jtu.tree_leaves(
            jax.tree_map(lambda a, b: jnp.allclose(a, b, rtol=rtol, atol=atol), a, b)
        )
    )


def batch_concat(tree: PyTree, num_batch_dims: int = 1) -> jnp.ndarray:
    """Flatten and concatenate nested array structure, keeping batch dims."""
    flatten_fn = lambda x: _flatten(x, num_batch_dims)
    flat_leaves = jax.tree_map(flatten_fn, tree)
    return jnp.concatenate(jax.tree_util.tree_leaves(flat_leaves), axis=-1)


def tree_zeros_like(tree: PyTree, dtype=None) -> PyTree[jnp.ndarray]:
    return jax.tree_map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), tree)


def tree_ones_like(tree: PyTree, dtype=None) -> PyTree[jnp.ndarray]:
    return jax.tree_map(lambda x: jnp.ones(x.shape, dtype or x.dtype), tree)


def tree_bools_like(tree, where=None, invert=False):

    t, f = (True, False) if not invert else (False, True)
    default_tree = jax.tree_util.tree_map(lambda _: t, tree)
    if where:
        return tree_at(where, default_tree, f)
    else:
        return default_tree


def tree_insert_IMPURE(tree, subtree, batch_idxs: tuple[int, ...]):
    def insert(a1, a2):
        a1[batch_idxs] = a2
        return a1

    jax.tree_util.tree_map(insert, tree, subtree)


def tree_batch(
    trees: Sequence, along_existing_first_axis: bool = False, backend: str = "numpy"
):
    jp = {"jax": jnp, "numpy": np}[backend]

    # otherwise scalar-arrays will lead to indexing error
    trees = jax.tree_map(lambda arr: jp.atleast_1d(arr), trees)

    if not along_existing_first_axis:
        trees = jax.tree_util.tree_map(lambda arr: arr[None], trees)

    if len(trees) == 0:
        return trees
    if len(trees) == 1:
        return trees[0]

    return jax.tree_util.tree_map(lambda *arrs: jp.concatenate(arrs, axis=0), *trees)


def tree_concat(
    trees: Sequence,
    along_existing_first_axis=False,
    backend="numpy",
    suppress_warning=False,
):
    if not suppress_warning:
        print("Warning: Deprecated (because too slow). Use `tree_batch` instead.")
    if backend == "jax":
        concat = jnp.concatenate
        atleast_1d = jnp.atleast_1d
    else:
        concat = np.concatenate
        atleast_1d = np.atleast_1d

    # otherwise scalar-arrays will lead to indexing error
    trees = jax.tree_map(lambda arr: atleast_1d(arr), trees)

    if len(trees) == 0:
        return trees
    if len(trees) == 1:
        if along_existing_first_axis:
            return trees[0]
        else:
            return jax.tree_util.tree_map(lambda arr: arr[None], trees[0])

    if along_existing_first_axis:
        sl = (slice(None),)
    else:
        sl = (
            None,
            slice(None),
        )

    initial = jax.tree_map(
        lambda a1, a2: concat((a1[sl], a2[sl]), axis=0), trees[0], trees[1]
    )
    stack = reduce(
        lambda tree1, tree2: jax.tree_map(
            lambda a1, a2: concat((a1, a2[sl]), axis=0), tree1, tree2
        ),
        trees[2:],
        initial,
    )
    return stack


def tree_shape(tree, axis: int = 0):
    return jtu.tree_flatten(tree)[0][0].shape[axis]


@partial(jax.jit, static_argnums=(2, 3, 4))
def tree_slice(tree, start, slice_size=1, axis=0, keepdim=False):
    def slicing_fun(arr):

        if slice_size > 1:
            return jax.lax.dynamic_slice_in_dim(
                arr, start_index=start, slice_size=slice_size, axis=axis
            )
        else:
            return jax.lax.dynamic_index_in_dim(
                arr, index=start, axis=axis, keepdims=keepdim
            )

    return jax.tree_util.tree_map(slicing_fun, tree)


@partial(jax.jit, static_argnums=(2,))
def tree_indices(tree, indices: jnp.ndarray, axis=0):
    """Extract an array of indices in an axis for every tree-element

    Args:
        tree (_type_): Tree of Arrays
        indices (jnp.ndarray): Array of Integers
        axis (int, optional): _description_. Defaults to 0.
    """

    def extract_indices_of_axis(arr):
        return jax.vmap(
            lambda index: jax.lax.dynamic_index_in_dim(
                arr, index, axis, keepdims=False
            ),
            out_axes=axis,
        )(indices)

    return jtu.tree_map(extract_indices_of_axis, tree)


# Delete then rnno/tree.py
def tree_split(tree, num_splits: int, axis: int = 0, lazy=True):
    """tree can *not* contain a list. It will silently break!!!"""  # TODO
    tree = jtu.tree_map(lambda arr: jnp.split(arr, num_splits, axis), tree)

    def get_loop_element(tree, i):
        return jtu.tree_map(
            lambda arr: arr[i], tree, is_leaf=lambda leaf: isinstance(leaf, list)
        )

    if lazy:
        for i in range(num_splits):
            yield get_loop_element(tree, i)
    else:
        return [get_loop_element(tree, i) for i in range(num_splits)]
