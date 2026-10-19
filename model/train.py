"""
train.py - train the 784-128-64-10 MLP in NumPy (forward and backward pass
written by hand, Adam optimizer) and save the float weights.

    python model/train.py

Output: model/float_model.npz
"""
import os
import time

import numpy as np

import mnist

LAYERS = [784, 128, 64, 10]
EPOCHS = 20
BATCH = 128
LR = 1e-3
LR_DECAY = 0.85          # learning rate multiplied by this after every epoch
WEIGHT_DECAY = 1e-4
SEED = 1

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(HERE, "float_model.npz")


def init_params(rng):
    params = []
    for fan_in, fan_out in zip(LAYERS[:-1], LAYERS[1:]):
        w = rng.standard_normal((fan_out, fan_in)).astype(np.float32) * np.sqrt(2.0 / fan_in)
        b = np.zeros(fan_out, dtype=np.float32)
        params += [w, b]
    return params


def forward(params, x):
    """Return the list of layer inputs and the final logits."""
    acts = [x]
    h = x
    n_layers = len(params) // 2
    for l in range(n_layers):
        w, b = params[2 * l], params[2 * l + 1]
        h = h @ w.T + b
        if l < n_layers - 1:
            h = np.maximum(h, 0.0)
        acts.append(h)
    return acts


def loss_and_grads(params, x, y):
    acts = forward(params, x)
    logits = acts[-1]
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=1, keepdims=True)
    n = x.shape[0]
    loss = -np.log(p[np.arange(n), y] + 1e-12).mean()

    grads = [None] * len(params)
    d = p
    d[np.arange(n), y] -= 1.0
    d /= n
    n_layers = len(params) // 2
    for l in reversed(range(n_layers)):
        w = params[2 * l]
        a_in = acts[l]
        grads[2 * l] = d.T @ a_in + WEIGHT_DECAY * w
        grads[2 * l + 1] = d.sum(axis=0)
        if l > 0:
            d = (d @ w) * (acts[l] > 0)
    return loss, grads


def accuracy(params, x, y):
    return float((forward(params, x)[-1].argmax(axis=1) == y).mean())


def main():
    rng = np.random.default_rng(SEED)
    tx, ty, vx, vy = mnist.load()
    tx = tx.astype(np.float32) / 255.0
    vx = vx.astype(np.float32) / 255.0

    params = init_params(rng)
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    step = 0
    lr = LR

    for epoch in range(EPOCHS):
        start = time.time()
        order = rng.permutation(len(tx))
        total = 0.0
        for i in range(0, len(tx), BATCH):
            idx = order[i:i + BATCH]
            loss, grads = loss_and_grads(params, tx[idx], ty[idx])
            total += loss * len(idx)
            step += 1
            for k in range(len(params)):
                m[k] = beta1 * m[k] + (1 - beta1) * grads[k]
                v[k] = beta2 * v[k] + (1 - beta2) * grads[k] ** 2
                m_hat = m[k] / (1 - beta1 ** step)
                v_hat = v[k] / (1 - beta2 ** step)
                params[k] -= (lr * m_hat / (np.sqrt(v_hat) + eps)).astype(np.float32)
        lr *= LR_DECAY
        print("epoch {:2d}  loss {:.4f}  test accuracy {:.2f}%  ({:.1f}s)".format(
            epoch + 1, total / len(tx), 100 * accuracy(params, vx, vy), time.time() - start))

    np.savez(OUT_FILE, **{"w{}".format(l): params[2 * l] for l in range(len(params) // 2)},
             **{"b{}".format(l): params[2 * l + 1] for l in range(len(params) // 2)})
    print("Float test accuracy: {:.2f}%".format(100 * accuracy(params, vx, vy)))
    print("Saved " + OUT_FILE)


if __name__ == "__main__":
    main()
