import os
import pathlib
import jax
import jax.numpy as jnp
from functools import partial
from jax import vmap, pmap
from jax.scipy.special import logsumexp


def random_split_like_tree(rng_key, target=None, treedef=None):
    if treedef is None: treedef = jax.tree_util.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


def normal_like_tree(rng_key, target, mean=0, std=1):
    # https://github.com/google/jax/discussions/9508#discussioncomment-2144076
    keys_tree = random_split_like_tree(rng_key, target)
    return jax.tree_map(
        lambda l, k: mean + std*jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


def generalized_logdet(m):
    singular_values = jnp.linalg.svd(m, compute_uv=False)
    return jnp.log(singular_values).sum()


def all_shards_close(x):
    return all(jnp.allclose(x[0], x[1]) for i in range(1, len(x)))


def split_into_batches(*arrays, n=None, bs=None):
    """
    - n: number of batches
    - bs: batch size
    Either `n` or `bs` has to be specified.
    """

    # compute num. of batches or batch size based on the other parameter
    assert (n is None) ^ (bs is None)
    bs = bs or len(arrays[0]) // n
    n = n or len(arrays[0]) // bs

    # batch data
    return [x[:n*bs].reshape([n, bs, *x.shape[1:]]) for x in arrays]


def subsample_dataset(key, x, y, n_class=10, samples_per_class=1000):
    """keep only `samples_per_class` images per class"""
    y_shuffled = jax.random.permutation(key, y)
    idx = jnp.concatenate([jnp.where(y_shuffled == i)[0][:samples_per_class] for i in range(n_class)])
    return x[idx], y[idx]


def eval_pmap(fn, x, y, batch_size=None, reduction=None):

    # function that aggregates outputs across devices and batches
    if reduction==None: reduce_fn = jnp.concatenate
    if reduction=='sum': reduce_fn = jnp.sum
    if reduction=='mean': reduce_fn = jnp.mean

    @partial(pmap, axis_name='i')
    def spmd_fn(x, y):

        # if batch size is None, evaluate whole dataset in parallel
        if batch_size is None: return fn(x, y)

        # otherwise, map over batches
        x_batched, y_batched = split_into_batches(x, y, bs=batch_size)
        batch_idxs = jnp.arange(len(x_batched))
        batch_fn = lambda i: fn(x_batched[i], y_batched[i])
        return reduce_fn(jax.lax.map(batch_fn, batch_idxs))

    # run spmd
    return reduce_fn(spmd_fn(x_sharded, y_sharded))


def predict_point(params, predict_fn, x):
    n_dev = jax.device_count()
    x_sharded = split_into_batches(x, n=n_dev)[0]
    params = pmap(lambda _: params)(jnp.arange(n_dev))
    logits = pmap(predict_fn)(x_sharded, params)
    return jnp.concatenate(logits)


def adv_attack(params, log_likelihood_fn, x, y, batch_size):

    def batch_fn(x, y):
        loss_fn = lambda x, y: log_likelihood_fn(params, x[None], y[None])
        return vmap(jax.jacfwd(loss_fn))(x, y)

    return eval_pmap(batch_fn, x, y, batch_size=batch_size)


def average_predictions(logits):
    chain_len, ds_len, n_classes = logits.shape
    logprobs = jax.nn.log_softmax(logits, 2) # map logits -> class probs
    logprobs = logsumexp(logprobs, 0) - jnp.log(chain_len) # average predictions
    return logprobs


def save_model(chain, logits_train, logits_test, val_hist, model_dir):
    pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
    jnp.save(f'{model_dir}/chain.npy', chain)
    jnp.save(f'{model_dir}/logits_train.npy', logits_train)
    jnp.save(f'{model_dir}/logits_test.npy', logits_test)
    jnp.save(f'{model_dir}/val_hist.npy', val_hist)
