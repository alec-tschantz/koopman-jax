from jax import Array, numpy as jnp, random as jr
import equinox as eqx


class Encoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear
    alpha: int
    input_dim: int
    latent_dim: int

    def __init__(
        self,
        input_rows: int,
        input_cols: int,
        latent_dim: int,
        alpha: int = 1,
        *,
        key: jr.PRNGKey,
    ):
        self.input_dim = input_rows * input_cols
        self.latent_dim = latent_dim
        self.alpha = alpha
        key1, key2, key3 = jr.split(key, 3)
        self.fc1 = eqx.nn.Linear(self.input_dim, 16 * alpha, key=key1)
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
    output_rows: int
    output_cols: int
    latent_dim: int
    alpha: int

    def __init__(
        self,
        output_rows: int,
        output_cols: int,
        latent_dim: int,
        alpha: int = 1,
        *,
        key: jr.PRNGKey,
    ):
        self.output_rows = output_rows
        self.output_cols = output_cols
        self.latent_dim = latent_dim
        self.alpha = alpha
        key1, key2, key3 = jr.split(key, 3)
        self.fc1 = eqx.nn.Linear(latent_dim, 16 * alpha, key=key1)
        self.fc2 = eqx.nn.Linear(16 * alpha, 16 * alpha, key=key2)
        self.fc3 = eqx.nn.Linear(16 * alpha, output_rows * output_cols, key=key3)

    def __call__(self, x: Array) -> Array:
        x = jnp.tanh(self.fc1(x))
        x = jnp.tanh(self.fc2(x))
        x = jnp.tanh(self.fc3(x))
        return x.reshape(self.output_rows, self.output_cols)


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


class KoopmanModel(eqx.Module):
    encoder: Encoder
    dynamics: Dynamics
    inverse_dynamics: InverseDynamics
    decoder: Decoder
    num_steps: int
    num_back_steps: int

    def __init__(
        self,
        input_rows: int,
        input_cols: int,
        latent_dim: int,
        num_steps: int,
        num_back_steps: int,
        alpha: int = 1,
        init_scale: float = 1.0,
        *,
        key: jr.PRNGKey,
    ):
        key_enc, key_dyn, key_dec, key_inv = jr.split(key, 4)
        self.encoder = Encoder(input_rows, input_cols, latent_dim, alpha, key=key_enc)
        self.dynamics = Dynamics(latent_dim, init_scale, key=key_dyn)
        self.inverse_dynamics = InverseDynamics(latent_dim, self.dynamics, key=key_inv)
        self.decoder = Decoder(input_rows, input_cols, latent_dim, alpha, key=key_dec)
        self.num_steps = num_steps
        self.num_back_steps = num_back_steps

    def __call__(self, x: Array, mode: str = "forward"):
        predictions = []
        back_predictions = []
        z = self.encoder(x)
        q = z
        if mode == "forward":
            for _ in range(self.num_steps):
                q = self.dynamics(q)
                predictions.append(self.decoder(q))
            predictions.append(self.decoder(z))
            return tuple(predictions), tuple(back_predictions)
        if mode == "backward":
            for _ in range(self.num_back_steps):
                q = self.inverse_dynamics(q)
                back_predictions.append(self.decoder(q))
            back_predictions.append(self.decoder(z))
            return tuple(predictions), tuple(back_predictions)
        return tuple(predictions), tuple(back_predictions)
