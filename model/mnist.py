"""
mnist.py - load the MNIST handwritten-digit dataset (IDX format).

The four .gz files are expected in model/data/. If they are missing they are
downloaded from a public mirror and checked against the official MD5 sums.
"""
import gzip
import hashlib
import os
import urllib.request

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    "test_images":  ("t10k-images-idx3-ubyte.gz",  "9fb629c4189551a2d022fa330f9573f3"),
    "test_labels":  ("t10k-labels-idx1-ubyte.gz",  "ec29112dd5afa0611ce80d1b7f02629c"),
}

MIRRORS = [
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _ensure(name):
    fname, md5 = FILES[name]
    path = os.path.join(DATA_DIR, fname)
    if os.path.exists(path) and _md5(path) == md5:
        return path
    os.makedirs(DATA_DIR, exist_ok=True)
    for mirror in MIRRORS:
        try:
            print("Downloading " + mirror + fname)
            urllib.request.urlretrieve(mirror + fname, path)
            if _md5(path) == md5:
                return path
        except OSError as err:
            print("  failed: {}".format(err))
    raise RuntimeError("Could not download " + fname)


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


def load_test():
    """Return (test_x, test_y) only; images as uint8 [10000, 784]."""
    vx = _read_idx(_ensure("test_images")).reshape(-1, 784)
    vy = _read_idx(_ensure("test_labels")).astype(np.int64)
    return vx, vy


def load():
    """Return (train_x, train_y, test_x, test_y); images as uint8 [n, 784]."""
    tx = _read_idx(_ensure("train_images")).reshape(-1, 784)
    ty = _read_idx(_ensure("train_labels")).astype(np.int64)
    vx = _read_idx(_ensure("test_images")).reshape(-1, 784)
    vy = _read_idx(_ensure("test_labels")).astype(np.int64)
    return tx, ty, vx, vy
