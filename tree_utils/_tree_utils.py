from functools import partial
from functools import reduce
from typing import Generic, Sequence, TypeVar

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import tree as tree_lib

T = TypeVar("T")


class PyTree(Generic[T]):
    pass

    def __iter__(self):
        pass

    def __next__(self):
        pass


def add_batch_dim(values: PyTree) -> PyTree[jax.Array]:
    return jax.tree.map(lambda x: jnp.expand_dims(x, axis=0), values)


def _flatten(x: jax.Array, num_batch_dims: int) -> jax.Array:
    if x.ndim < num_batch_dims:
        return x
    return jnp.reshape(x, list(x.shape[:num_batch_dims]) + [-1])


def tree_close(a: PyTree, b: PyTree, rtol=1e-05, atol=1e-08) -> bool:
    return all(
        jtu.tree_leaves(
            jax.tree.map(lambda a, b: jnp.allclose(a, b, rtol=rtol, atol=atol), a, b)
        )
    )


def batch_concat_acme(
    tree: PyTree,
    num_batch_dims: int = 1,
) -> jax.Array:
    """Flatten and concatenate nested array structure, keeping batch dims.
    IGNORES the ordered of elements in an `OrderedDict`, see EngineeringLog @ 18.02.23
    """
    flatten_fn = lambda x: _flatten(x, num_batch_dims)
    flat_leaves = tree_lib.map_structure(flatten_fn, tree)
    return jnp.concatenate(tree_lib.flatten(flat_leaves), axis=-1)


def batch_concat(tree: PyTree, num_batch_dims: int = 1) -> jax.Array:
    """Flatten and concatenate nested array structure, keeping batch dims."""
    flatten_fn = lambda x: _flatten(x, num_batch_dims)
    flat_leaves = jax.tree.map(flatten_fn, tree)
    return jnp.concatenate(jax.tree_util.tree_leaves(flat_leaves), axis=-1)


def tree_zeros_like(tree: PyTree, dtype=None) -> PyTree[jax.Array]:
    return jax.tree.map(lambda x: jnp.zeros(x.shape, dtype or x.dtype), tree)


def tree_ones_like(tree: PyTree, dtype=None) -> PyTree[jax.Array]:
    return jax.tree.map(lambda x: jnp.ones(x.shape, dtype or x.dtype), tree)


def tree_bools_like(tree, where=None, invert=False):
    from equinox import tree_at

    t, f = (True, False) if not invert else (False, True)
    default_tree = jax.tree.map(lambda _: t, tree)
    if where:
        return tree_at(where, default_tree, f)
    else:
        return default_tree


def tree_insert_IMPURE(tree, subtree, batch_idxs: tuple[int, ...]):
    def insert(a1, a2):
        a1[batch_idxs] = a2
        return a1

    jax.tree.map(insert, tree, subtree)


def is_jax_or_numpy_pytree(tree: PyTree) -> str:
    flat_leaves = lambda obj: jtu.tree_flatten(
        jax.tree.map(lambda arr: isinstance(arr, obj), tree)
    )[0]
    is_numpy = flat_leaves(np.ndarray)
    if all(is_numpy):
        return "numpy"
    is_jax = flat_leaves(jax.Array)
    if all(is_jax):
        return "jax"
    raise Exception("Not all leaves are either jax.Arrays or numpy.ndarrays")


def tree_batch(
    trees: Sequence,
    along_existing_first_axis: bool = False,
    backend: str | None = "numpy",
):
    if backend is None:
        backend = is_jax_or_numpy_pytree(trees)
    jp = {"jax": jnp, "numpy": np}[backend]

    if not along_existing_first_axis:
        # convert all IntEnums -> jax.Array
        # as IntEnums are not subscriptable
        trees = jax.tree.map(jp.asarray, trees)

        trees = jax.tree.map(lambda arr: arr[None], trees)
    else:
        # otherwise scalar-arrays will lead to indexing error
        trees = jax.tree.map(lambda arr: jp.atleast_1d(arr), trees)

    if len(trees) == 0:
        return trees
    if len(trees) == 1:
        return trees[0]

    return jax.tree.map(lambda *arrs: jp.concatenate(arrs, axis=0), *trees)


def tree_concat(
    trees: Sequence,
    along_existing_first_axis=False,
    backend="numpy",
    suppress_warning=False,
):
    if not suppress_warning:
        print(
            "Warning: Deprecated `tree_concat`\
                  (because it is too slow). Use `tree_batch` instead."
        )
    if backend == "jax":
        concat = jnp.concatenate
        atleast_1d = jnp.atleast_1d
    else:
        concat = np.concatenate
        atleast_1d = np.atleast_1d

    # otherwise scalar-arrays will lead to indexing error
    trees = jax.tree.map(lambda arr: atleast_1d(arr), trees)

    if len(trees) == 0:
        return trees
    if len(trees) == 1:
        if along_existing_first_axis:
            return trees[0]
        else:
            return jax.tree.map(lambda arr: arr[None], trees[0])

    if along_existing_first_axis:
        sl = (slice(None),)
    else:
        sl = (
            None,
            slice(None),
        )

    initial = jax.tree.map(
        lambda a1, a2: concat((a1[sl], a2[sl]), axis=0), trees[0], trees[1]
    )
    stack = reduce(
        lambda tree1, tree2: jax.tree.map(
            lambda a1, a2: concat((a1, a2[sl]), axis=0), tree1, tree2
        ),
        trees[2:],
        initial,
    )
    return stack


def tree_shape(tree, axis: int = 0) -> int:
    return jtu.tree_flatten(tree)[0][0].shape[axis]


def tree_ndim(tree) -> int:
    return jtu.tree_flatten(tree)[0][0].ndim


def tree_map_flat(tree, f, *args):
    """Maps `f(arr, *args)` across a flattened and concatenated version of `tree`."""
    arr, unflatten = ravel_pytree(tree)
    arr = f(arr, *args)
    return unflatten(arr)


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

    return jax.tree.map(slicing_fun, tree)


@partial(jax.jit, static_argnums=(2,))
def tree_indices(tree, indices: jax.Array, axis=0):
    """Extract an array of indices in an axis for every tree-element

    Args:
        tree (_type_): Tree of Arrays
        indices (jax.Array): Array of Integers
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


def tree_standardize(tree, axes=None, eps=1e-8):
    ndim = tree_ndim(tree)
    assert ndim > 1, "Expected last axis to be features, thus must be at least 2d"

    def standardizer(arr):
        nonlocal axes
        if axes is None:
            axes = tuple(range(ndim - 1))
        return (arr - jnp.mean(arr, axis=axes)) / (jnp.std(arr, axis=axes) + eps)

    return jax.tree.map(standardizer, tree)


def to_3d_if_2d(tree, strict: bool = False):
    ndim = tree_ndim(tree)
    assert ndim == 2 or ndim == 3
    if strict:
        assert ndim == 2
    if ndim == 2:
        return add_batch_dim(tree)
    return tree


def to_2d_if_3d(tree, idx: int = 0, strict: bool = False):
    ndim = tree_ndim(tree)
    assert ndim == 2 or ndim == 3
    if strict:
        assert ndim == 3
    if ndim == 3:
        return tree_slice(tree, idx)
    return tree
