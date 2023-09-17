from jax import config
config.update('jax_enable_x64', False)

# simulate 8 devices (like a TPU-v3-8)
# import os
# flags = os.environ.get('XLA_FLAGS', '')
# os.environ['XLA_FLAGS'] = flags + ' --xla_force_host_platform_device_count=8'

from run import run

"""
This script allows simple local testing/debugging of the 'run.py' script.
"""

if __name__ == "__main__":
    run(model_name = 'cnn',
        ds_name = 'mnist',
        samples_per_class = 100,
        distribution = 'dirichlet-jac',
        distribution_param = 0.01,
        normal_prior_scale = 1,
        posterior = False,
        T = 0,
        augment = False,
        lr = 1e-4,
        batch_size = 100,
        n_epochs = 1_000,
        seed = 0,
        save = False,
    )
