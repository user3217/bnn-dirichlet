import jax
import jax.numpy as jnp
import operator as op
from jax import vmap
from jax.tree_util import tree_map, tree_leaves, tree_reduce
from typing import Optional
import utils


def add_normal_prior(posterior_fn, std, ds_size):

    def out_fn(params, x, y):
        
        # get original posterior
        log_posterior, logits = posterior_fn(params, x, y)
        batch_size = len(logits)

        # add normal prior rescaled to batch size
        # the sum of this prior over batches equals the full-batch prior
        if std is not None:
            dx = tree_reduce(op.add, tree_map(lambda x: (x**2).sum(), params))
            log_prior = -0.5 * dx / std**2
            log_prior *= batch_size / ds_size
            log_posterior += log_prior

        return log_posterior, logits

    return out_fn


def make_categorical_pdf(predict_fn):

    def out_fn(params, x, y):
        logits, _ = predict_fn(x, params)
        batch_size, n_class = logits.shape
        y_one_hot = jax.nn.one_hot(y, n_class)
        log_prob = (y_one_hot * jax.nn.log_softmax(logits)).sum()
        return log_prob, logits

    return out_fn


def make_dirichlet_pdf(predict_fn, alpha_prior, posterior=True, correction=None):
    
    def out_fn(params, x, y):

        # predict
        logits, model_correction = predict_fn(x, params)
        batch_size, n_class = logits.shape

        # optionally add correction (logits -> probs)
        alpha = alpha_prior * jnp.ones(n_class)
        if correction is not None: alpha += 1 # 

        # get posterior
        if posterior: alpha += jax.nn.one_hot(y, n_class)
        log_probs = jax.nn.log_softmax(logits)
        log_prob = ((alpha-1) * log_probs).sum()

        # optionally add correction (params -> logits)
        if correction == 'mat':
            for layer_name, layer_data in params.items():
                assert layer_name.startswith('linear'), f"'full' correction only supports linear layers, got {layer_name}"
                model_correction += utils.generalized_logdet(layer_data['w'])
            log_prob += model_correction
        if correction == 'jac':
            def model_correction_per_sample(x):
                predict_fn_single = lambda params: predict_fn(x[None], params)[0][0]
                jac = jax.jacrev(predict_fn_single)(params)
                jac = jnp.concatenate(tree_map(lambda x: x.reshape(n_class, -1), tree_leaves(jac)), 1)
                singular_values = jnp.linalg.svd(jac, compute_uv=False)
                model_correction = jnp.log(singular_values).sum()
                return model_correction
            log_prob += vmap(model_correction_per_sample)(x).sum()

        return log_prob, logits

    return out_fn


def make_dir_gauss_post_fn(predict_fn, alpha_prior, from_logprobs=False):
    """
    Gaussian approximation of the Dirichlet posterior.
    """

    def out_fn(params, x, y):

        # predict
        logits, _ = predict_fn(x, params)
        batch_size, n_class = logits.shape

        # the guassian approximation can either be applied over logits (defualt)
        # or over logprobs, which is just logits shifted by a scalar value
        if from_logprobs: logits = jax.nn.log_softmax(logits)

        # get posterior
        y_one_hot = jax.nn.one_hot(y, n_class)
        alpha_post = alpha_prior + y_one_hot # [batch_size, n_class]
        var = jnp.log(1 / alpha_post + 1) # [batch_size, n_class]
        mean = jnp.log(alpha_post) - var / 2 # [batch_size, n_class]
        if from_logprobs: mean -= mean.max() # logprobs are (-inf, 0)
        log_posterior = (-0.5 * (logits - mean)**2 / var).sum()
        
        return log_posterior, logits

    return out_fn


def make_input_sensitivity_pdf(predict_fn, prior_scale, posterior=True):

    def out_fn(params, x, y):

        # get prior
        def predict_single_max(x):
            logits, _ = predict_fn(x[None], params)[0]
            max_logit = logits.max()
            return max_logit, logits
        def norm_single(x):
            grads, logits = jax.grad(predict_single_max, has_aux=True)(x)
            sensitivity = jnp.abs(grads).mean()
            return sensitivity, logits
        sensitivity, logits = vmap(norm_single)(x)
        log_prob = prior_scale*sensitivity.sum()

        # optionally add likelihood
        if posterior:
            batch_size, n_class = logits.shape
            y_one_hot = jax.nn.one_hot(y, n_class)
            categ_likelihood = jnp.sum(y_one_hot * jax.nn.log_softmax(logits))
            log_prob += categ_likelihood

        return log_prob, logits

    return out_fn
