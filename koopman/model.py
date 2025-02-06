import equinox as eqx
from jax import Array, numpy as jnp, random as jr, lax


class Encoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        alpha: int = 1,
        *,
        key: jr.PRNGKey,
    ):
        key1, key2, key3 = jr.split(key, 3)
        self.fc1 = eqx.nn.Linear(input_dim, 16 * alpha, key=key1)
        self.fc2 = eqx.nn.Linear(16 * alpha, 16 * alpha, key=key2)
        self.fc3 = eqx.nn.Linear(16 * alpha, latent_dim, key=key3)

    def __call__(self, x: Array) -> Array:
        x = jnp.tanh(self.fc1(x))
        x = jnp.tanh(self.fc2(x))
        return self.fc3(x)


class Decoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        alpha: int = 1,
        *,
        key: jr.PRNGKey,
    ):
        key1, key2, key3 = jr.split(key, 3)
        self.fc1 = eqx.nn.Linear(latent_dim, 16 * alpha, key=key1)
        self.fc2 = eqx.nn.Linear(16 * alpha, 16 * alpha, key=key2)
        self.fc3 = eqx.nn.Linear(16 * alpha, output_dim, key=key3)

    def __call__(self, x: Array) -> Array:
        x = jnp.tanh(self.fc1(x))
        x = jnp.tanh(self.fc2(x))
        x = jnp.tanh(self.fc3(x))
        return x


class Dynamics(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, init_scale: float, *, key: jr.PRNGKey):
        key_linear, key_gauss = jr.split(key)
        weight = jr.normal(key_gauss, (latent_dim, latent_dim)) / latent_dim
        u, _, vh = jnp.linalg.svd(weight, full_matrices=False)
        weight = (u @ vh) * init_scale
        self.linear = eqx.nn.Linear(
            latent_dim, latent_dim, use_bias=False, key=key_linear
        )
        self.linear = eqx.tree_at(lambda l: l.weight, self.linear, weight)

    def __call__(self, x: Array) -> Array:
        return self.linear(x)


class InverseDynamics(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self, latent_dim: int, dynamics: Dynamics, *, key: jr.PRNGKey):
        key_linear = key
        inv_weight = jnp.linalg.pinv(dynamics.linear.weight.T)
        self.linear = eqx.nn.Linear(
            latent_dim, latent_dim, use_bias=False, key=key_linear
        )
        self.linear = eqx.tree_at(lambda l: l.weight, self.linear, inv_weight)

    def __call__(self, x: Array) -> Array:
        return self.linear(x)


class Koopman(eqx.Module):
    encoder: Encoder
    decoder: Decoder
    dynamics: Dynamics
    inverse_dynamics: InverseDynamics

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        alpha: int = 1,
        init_scale: float = 0.99,
        *,
        key: jr.PRNGKey,
    ):
        key_enc, key_dyn, key_dec, key_inv = jr.split(key, 4)
        self.encoder = Encoder(input_dim, latent_dim, alpha, key=key_enc)
        self.dynamics = Dynamics(latent_dim, init_scale, key=key_dyn)
        self.inverse_dynamics = InverseDynamics(latent_dim, self.dynamics, key=key_inv)
        self.decoder = Decoder(latent_dim, input_dim, alpha, key=key_dec)

    def forward(self, x: Array, num_steps: int):
        def step(z, _):
            z = self.dynamics(z)
            return z, self.decoder(z)

        z0 = self.encoder(x)
        x0 = self.decoder(z0)

        _, xs = lax.scan(step, z0, None, length=num_steps)
        return jnp.concatenate([x0[None], xs], axis=0)


    def backward(self, x: Array, num_steps: int):
        def step(z, _):
            z = self.inverse_dynamics(z)
            return z, self.decoder(z)

        z0 = self.encoder(x)
        x0 = self.decoder(z0)

        _, xs = lax.scan(step, z0, None, length=num_steps)
        return jnp.concatenate([x0[None], xs], axis=0)
