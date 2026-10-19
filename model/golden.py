"""
golden.py - bit-exact integer reference model of the TinyNPU hardware.

Every operation here matches the RTL exactly:
  input       x0 = pixel >> 1                       (uint8 pixel -> int8 0..127)
  layer l     acc = sum_k W[n][k] * x[k] + B[n]     (int8 x int8, int32 accumulate)
  requantize  y = clamp((acc * M + 2^(S-1)) >>> S)  (round half up, arithmetic shift)
              clamp to [0, 127] after ReLU (hidden layers), [-128, 127] for logits
  prediction  index of the largest logit (lowest index wins a tie)
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
QUANT_FILE = os.path.join(HERE, "quant_model.npz")


def load_quant_model(path=QUANT_FILE):
    """Return a list of layer dicts: W (int64 [out,in]), B, M, S, relu."""
    z = np.load(path)
    layers = []
    for l in range(int(z["n_layers"])):
        layers.append({
            "W": z["W{}".format(l)].astype(np.int64),
            "B": z["B{}".format(l)].astype(np.int64),
            "M": int(z["M{}".format(l)]),
            "S": int(z["S{}".format(l)]),
            "relu": bool(z["relu{}".format(l)]),
        })
    return layers


def quantize_input(pixels):
    """uint8 pixels -> int8 activations in 0..127 (same as the hardware's >>1)."""
    return (np.asarray(pixels, dtype=np.int64) >> 1)


def requant(acc, m, s, relu):
    """Integer rescale exactly as the requantization pipeline in the RTL."""
    acc = np.asarray(acc, dtype=np.int64)
    y = (acc * m + (1 << (s - 1))) >> s
    lo = 0 if relu else -128
    return np.clip(y, lo, 127)


def infer(layers, pixels):
    """
    pixels: uint8 [n_images, 784]
    returns (logits int64 [n_images, n_out], predictions int64 [n_images])
    """
    x = quantize_input(pixels)                   # [n, 784]
    for layer in layers:
        acc = x @ layer["W"].T + layer["B"]      # int64, never overflows int32 here
        x = requant(acc, layer["M"], layer["S"], layer["relu"])
    logits = x
    preds = np.argmax(logits, axis=1)            # first maximum wins, like the RTL
    return logits, preds
