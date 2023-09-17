import numpy as np
from itertools import product
from run import run
kwargs = dict(ds_name='mnist', model_name='cnn', n_epochs=10_000, lr=1e-4, samples_per_class=100)

# this code was run on TPU devices with 8 cores each, over 10 seeds, resulting in 80 posterior samples
for seed in range(10):
    for model in ('mnist', 'cnn'):

        # categorical baseline
        run(distribution='categorical', seed=seed, **kwargs)

        # dirichlet models
        distributions = ('dirichlet-jac', 'dirichlet-partial')
        alphas = np.geomspace(0.001, 0.999, 10)
        for alpha, distribution, posterior in product(distributions, alphas, (False, True))
            run(model_name=model, distribution=distribution, distribution_param=alpha, posterior=posterior, seed=seed, **kwargs)
