from jax import Array, numpy as jnp
import jax
import optax
import equinox as eqx

def mse_loss(pred: Array, target: Array) -> Array:
    return jnp.mean((pred - target) ** 2)

def train(
    model: eqx.Module,
    train_loader,
    lr: float,
    weight_decay: float,
    lamb: float,
    num_epochs: int,
    lr_decay: float,
    decay_epochs: list[int],
    nu: float = 0.0,
    eta: float = 0.0,
    backward: int = 0,
    num_steps: int = 1,
    num_back_steps: int = 1,
    grad_clip: float = 1.0,
):
    current_lr = lr
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(current_lr, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    epoch_history, loss_history = [], []

    @eqx.filter_jit
    def update_step(model, opt_state, batch):
        def loss_fn(model):
            v_forward = jax.vmap(lambda x: model(x, mode="forward"))
            forward_preds, _ = v_forward(batch[0])
            loss_forward = 0.0
            for k in range(num_steps):
                loss_forward += mse_loss(forward_preds[k], batch[k + 1])
            loss_identity = mse_loss(forward_preds[-1], batch[0]) * num_steps
            loss_backward = 0.0
            loss_consistency = 0.0
            if backward:
                v_backward = jax.vmap(lambda x: model(x, mode="backward"))
                _, back_preds = v_backward(batch[-1])
                for k in range(num_back_steps):
                    loss_backward += mse_loss(back_preds[k], batch[::-1][k + 1])
                A = model.dynamics.linear.weight
                B = model.inverse_dynamics.linear.weight
                latent_dim = A.shape[-1]
                for k in range(1, latent_dim + 1):
                    A_sub = A[:, :k]
                    B_sub = B[:k, :]
                    I_k = jnp.eye(k)
                    loss_consistency += (
                        jnp.sum((B_sub @ A_sub - I_k) ** 2)
                        + jnp.sum((A_sub @ B_sub - I_k) ** 2)
                    ) / (2.0 * k)
            return loss_forward + lamb * loss_identity + nu * loss_backward + eta * loss_consistency

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state_new = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state_new, loss

    for epoch in range(num_epochs):
        for batch in train_loader:
            model, opt_state, loss = update_step(model, opt_state, batch)
        if epoch in decay_epochs:
            current_lr *= lr_decay
        loss_history.append(loss)
        epoch_history.append(epoch)
    return model, optimizer, [epoch_history, loss_history[-1], 0.0]
