from jax import Array, numpy as jnp, vmap

import optax
import equinox as eqx
from tqdm import trange

import koopman

def mse_loss(pred: Array, target: Array) -> Array:
    return jnp.mean((pred - target) ** 2)


def forward_loss(model, batch, num_steps):
    preds = vmap(model.forward, in_axes=(0, None))(batch[0], num_steps)
    preds = preds.swapaxes(0, 1)
    return sum(mse_loss(preds[k], batch[k + 1]) for k in range(num_steps))


def identity_loss(model, batch, num_steps):
    preds = vmap(model.forward, in_axes=(0, None))(batch[0], num_steps)
    preds = preds.swapaxes(0, 1)
    return mse_loss(preds[0], batch[0]) * num_steps


def backward_loss(model, batch, num_steps):
    back_preds = vmap(model.backward, in_axes=(0, None))(batch[-1], num_steps)
    back_preds = back_preds.swapaxes(0, 1)
    return sum(mse_loss(back_preds[k], batch[::-1][k + 1]) for k in range(num_steps))


def consistency_loss(model):
    A, B = model.dynamics.linear.weight, model.inverse_dynamics.linear.weight
    latent_dim = A.shape[-1]
    return sum(
        (
            jnp.sum((B[:k, :] @ A[:, :k] - jnp.eye(k)) ** 2)
            + jnp.sum((A[:k, :] @ B[:, :k] - jnp.eye(k)) ** 2)
        )
        / (2.0 * k)
        for k in range(1, latent_dim + 1)
    )


def train(
    model: eqx.Module,
    x_train: Array,
    num_epochs: int = 600,
    batch_dim: int = 64,
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    forward_steps: int = 8,
    backward_steps: int = 8,
    beta_forward: float = 1.0,
    beta_backward: float = 1e-1,
    beta_identity: float = 1.0,
    beta_consistency: float = 1e-2,
    grad_clip: float = 1.0,
):
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def update_step(model, opt_state, batch):
        def loss_fn(model):
            return (
                beta_forward * forward_loss(model, batch, forward_steps)
                + beta_backward * identity_loss(model, batch, forward_steps)
                + beta_identity * backward_loss(model, batch, backward_steps)
                + beta_consistency * consistency_loss(model)
            )

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state_new, loss

    epoch_losses = []
    for epoch in trange(num_epochs, desc="Training"):
        total_loss = 0.0
        train_loader = list(koopman.train_loader(x_train, forward_steps, batch_dim))
        for batch in train_loader:
            model, opt_state, loss = update_step(model, opt_state, batch)
            total_loss = total_loss + loss
        epoch_losses.append(total_loss / len(train_loader))

    return model, epoch_losses
