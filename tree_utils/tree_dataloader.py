import functools as ft
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jrand

from tree_utils import PyTree, tree_indices, tree_shape


class MiniBatchState(NamedTuple):
    indices: jax.Array
    i: int
    bs: int
    n_minibatches: int
    minibatch_size: int
    key: jrand.PRNGKey


class TreeDataloader(NamedTuple):
    init: Callable[[PyTree], MiniBatchState]
    next: Callable[
        [MiniBatchState, PyTree],
        Tuple[MiniBatchState, PyTree],
    ]


def tree_dataloader(
    key: jrand.PRNGKey,
    n_minibatches: int = 1,
    axis: int = 0,
    reshuffle: bool = True,
    tree_transform: Optional[Callable] = None,
    do_bootstrapping: bool = False,
) -> TreeDataloader:
    """A utility that allows minibatching a PyTree along an axis.

    Args:
        key (jrand.PRNGKey): Seed of minibatching permutation.
        n_minibatches (int, optional): Number of minibatches. Defaults to 1.
        axis (int, optional): Axis along which to minibatch. Defaults to 0.
        reshuffle (bool, optional): Whether to reshuffle after one episode.
            Defaults to True.
        tree_transform (Optional[Callable], optional): A transform that is applied to
            the extracted minibatch of the PyTree. Defaults to None.
            Example: tree_transform(key, tree, minibatch_size) -> transformed_tree
        do_bootstrapping (bool, optional): Whether to do bootstrapping, this allows
            for a n_minibatches that does not divide the batchsize. Defaults to False.

    Returns:
        TreeDataloader: A function tuple of (init, next).
            Example:
                >>> state = init(tree)
                >>> state, minibatch = next(state, tree)
    """

    def init(tree: PyTree) -> MiniBatchState:
        bs = tree_shape(tree, axis)
        minibatch_size = _bootstrap_minibatch_size(n_minibatches, bs, do_bootstrapping)
        inner_key, consume = jrand.split(key)

        return MiniBatchState(
            _gen_minibatch_indices(consume, bs, n_minibatches, minibatch_size),
            0,
            bs,
            n_minibatches,
            minibatch_size,
            inner_key,
        )

    def next(state: MiniBatchState, tree: PyTree) -> Tuple[MiniBatchState, PyTree]:

        indices = state.indices
        key = state.key
        if state.i >= state.n_minibatches:
            # iteration over one epoch is done
            if reshuffle:
                key, consume = jrand.split(key)
                indices = _gen_minibatch_indices(
                    consume, state.bs, state.n_minibatches, state.minibatch_size
                )

        # reset counter if required
        i = state.i % state.n_minibatches

        batch_of_tree = tree_indices(tree, indices[i], axis)

        if tree_transform:
            key, consume = jrand.split(key)
            batch_of_tree = tree_transform(consume, batch_of_tree, state.minibatch_size)

        return (
            MiniBatchState(
                indices, i + 1, state.bs, state.n_minibatches, state.minibatch_size, key
            ),
            batch_of_tree,
        )

    return TreeDataloader(init, next)


@ft.partial(jax.jit, static_argnums=(1, 2, 3))
def _gen_minibatch_indices(
    key, batch_size: int, n_minibatches: int, minibatch_size: int
) -> jax.Array:
    consume1, consume2 = jrand.split(key)
    permutation = jax.random.permutation(consume1, jnp.arange(batch_size))
    permutation_bootstrap = jax.random.permutation(consume2, jnp.arange(batch_size))
    permutation = jnp.hstack((permutation, permutation_bootstrap))

    def scan_fn(carry, _):
        start_idx = carry
        y = jnp.take(
            jnp.arange(batch_size),
            jax.lax.dynamic_slice_in_dim(permutation, start_idx, minibatch_size),
        )
        carry = start_idx + minibatch_size
        return carry, y

    return jax.lax.scan(scan_fn, 0, length=n_minibatches, xs=None)[1]


@ft.partial(jax.jit, static_argnums=(1, 2, 3))
def _gen_minibatch_masks(
    key, batch_size: int, n_minibatches: int, minibatch_size: int
) -> jax.Array:

    idxss = _gen_minibatch_indices(key, batch_size, n_minibatches, minibatch_size)

    # generate masks from idxs
    def to_mask(idxs):
        return jnp.in1d(jnp.arange(batch_size), idxs)

    return jax.vmap(to_mask)(idxss)


def _bootstrap_minibatch_size(
    n_minibatches: int, batchsize: int, do_bootstrapping: bool
) -> int:
    if not do_bootstrapping:
        assert batchsize % n_minibatches == 0
    else:
        for i in range(1000):  # TODO
            batchsize += i
            if batchsize % n_minibatches == 0:
                break
        else:
            raise Exception("Impossible! :)")

    return batchsize // n_minibatches
