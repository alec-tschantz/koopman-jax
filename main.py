import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from jax import numpy as jnp, random as jr
import jax
from koopman.model import KoopmanModel
from koopman.train import train
from koopman.data import pendulum_data, add_channels, get_train_loader


def test_model(
    model, X_test: np.ndarray, pred_steps: int, m: int, n: int
) -> np.ndarray:
    errors = []
    num_tests = min(30, X_test.shape[0] - pred_steps)
    for i in range(num_tests):
        error_temp = []
        x0 = X_test[i].reshape(-1)
        latent = model.encoder(x0)
        for j in range(pred_steps):
            latent = model.dynamics(latent)
            x_pred = model.decoder(latent)
            target_sample = X_test[i + j + 1].reshape(m, n)
            rel_error = jnp.linalg.norm(x_pred - target_sample) / jnp.linalg.norm(
                target_sample
            )
            error_temp.append(rel_error)
        errors.append(np.array(error_temp))
    return np.array(errors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="test")
    parser.add_argument("--bottleneck", type=int, default=6)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--theta", type=float, default=2.4)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--steps_back", type=int, default=8)
    parser.add_argument("--lr_update", type=int, nargs="+", default=[30, 200, 400, 500])
    parser.add_argument("--lr_decay", type=float, default=0.2)
    parser.add_argument("--init_scale", type=float, default=0.99)
    parser.add_argument("--gradclip", type=float, default=0.05)
    parser.add_argument("--pred_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    backward = not args.baseline

    np.random.seed(args.seed)
    key = jr.PRNGKey(args.seed)

    if not os.path.exists(args.folder):
        os.makedirs(args.folder)

    X_train, X_test, X_train_clean, X_test_clean, m, n = pendulum_data(
        noise=args.noise, theta=args.theta
    )
    X_train = add_channels(X_train)
    X_test = add_channels(X_test)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    train_sequences = [
        X_train_flat[i:] if i == 0 else X_train_flat[:-i]
        for i in range(args.steps, -1, -1)
    ]
    train_loader = list(get_train_loader(train_sequences, args.batch))

    model = KoopmanModel(
        m,
        n,
        args.bottleneck,
        args.steps,
        args.steps_back,
        args.alpha,
        args.init_scale,
        key=key,
    )
    model, _, _ = train(
        model,
        train_loader,
        lr=args.lr,
        weight_decay=args.wd,
        lamb=args.bottleneck,
        num_epochs=args.epochs,
        lr_decay=args.lr_decay,
        decay_epochs=args.lr_update,
        nu=args.alpha,
        eta=args.steps,
        backward=int(backward),
        num_steps=args.steps,
        num_back_steps=args.steps_back,
        grad_clip=args.gradclip,
    )

    errors = test_model(model, X_test_flat, args.pred_steps, m, n)
    mean_errors = np.mean(errors, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(mean_errors, "o--", lw=2, color="#377eb8")
    plt.fill_between(
        np.arange(mean_errors.shape[0]),
        np.quantile(errors, 0.05, axis=0),
        np.quantile(errors, 0.95, axis=0),
        color="#377eb8",
        alpha=0.2,
    )
    plt.xlabel("Time step", fontsize=14)
    plt.ylabel("Relative prediction error", fontsize=14)
    plt.tight_layout()
    plt.show()
