import argparse
import os
import numpy as np
from jax import random as jr
from koopman.model import KoopmanModel
from koopman.train import train
from koopman.data import pendulum_data, get_train_loader

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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    backward = not args.baseline

    np.random.seed(args.seed)
    key = jr.PRNGKey(args.seed)

    if not os.path.exists(args.folder):
        os.makedirs(args.folder)

    x_train, x_test, x_train_clean, x_test_clean, input_dim = pendulum_data(
        noise=args.noise, theta=args.theta
    )
    train_sequences = [
        x_train[i:] if i == 0 else x_train[:-i] for i in range(args.steps, -1, -1)
    ]
    train_loader = list(get_train_loader(train_sequences, args.batch))

    model = KoopmanModel(
        input_dim,
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
