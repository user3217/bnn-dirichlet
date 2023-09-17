import jax
import jax.numpy as jnp


def image_grid(images, grid_width, grid_height):
    n, h, w, c = images.shape
    grid = images[:grid_width*grid_height] # discard images
    grid = grid.reshape([grid_width, grid_height, h, w, c])
    grid = jnp.transpose(grid, (1, 2, 0, 3, 4))
    grid = grid.reshape([grid_height*h, grid_width*w, c])
    return grid



def augment(key, images):
    """
    Matches the paper "How Good is the Bayes Posterior in Deep Neural Networks Really?" exactly.
    https://github.com/google-research/google-research/blob/fa49cd0231954458e2b02278d95528b70245f06d/cold_posterior_bnn/datasets.py#L113
    """
    n, h, w, c = images.shape
    key_flip, key_crop = jax.random.split(key)
    
    # left-right flip
    key_flip, key_crop = jax.random.split(key)
    do_flip = jax.random.bernoulli(key_flip, 0.5, [n])
    images = jnp.where(do_flip[:, None, None, None], jnp.flip(images, 2), images)
    
    # pad + crop
    images = jnp.pad(images, [[0, 0], [4, 4], [4, 4], [0, 0]])
    dx, dy = jax.random.randint(key_crop, [2], 0, 7)
    images = jax.lax.dynamic_slice(images, (0, dy, dx, 0), (n, h, w, c))

    return images
