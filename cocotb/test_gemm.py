"""
test_gemm.py - cocotb tests for npu_gemm (skew + systolic array + de-skew).

Random INT8 tiles of random length are streamed in, some back to back and
some with idle gaps, and every result vector is compared with numpy.
"""
import os
import random

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

N = int(os.environ.get("N", "4"))
ACC_W = 32


def pack(values, width):
    word = 0
    mask = (1 << width) - 1
    for i, v in enumerate(values):
        word |= (int(v) & mask) << (width * i)
    return word


def unpack_signed(word, width, count):
    out = []
    for i in range(count):
        v = (word >> (width * i)) & ((1 << width) - 1)
        if v >= 1 << (width - 1):
            v -= 1 << width
        out.append(v)
    return out


async def reset(dut):
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    dut.in_first.value = 0
    dut.in_last.value = 0
    dut.in_a.value = 0
    dut.in_b.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def collect(dut, results, expected_vectors):
    while len(results) < expected_vectors:
        await RisingEdge(dut.clk)
        await ReadOnly()
        if dut.out_valid.value == 1:
            results.append(unpack_signed(int(dut.out_res.value), ACC_W, N))


async def run_tiles(dut, tiles, gaps):
    """Stream the tiles; gaps[t] idle cycles are inserted before tile t."""
    results = []
    monitor = cocotb.start_soon(collect(dut, results, N * len(tiles)))
    for (a, b), gap in zip(tiles, gaps):
        for _ in range(gap):
            await RisingEdge(dut.clk)
            dut.in_valid.value = 0
            dut.in_first.value = 0
            dut.in_last.value = 0
        k_len = a.shape[1]
        for k in range(k_len):
            await RisingEdge(dut.clk)
            dut.in_valid.value = 1
            dut.in_first.value = int(k == 0)
            dut.in_last.value = int(k == k_len - 1)
            dut.in_a.value = pack(a[:, k], 8)
            dut.in_b.value = pack(b[k, :], 8)
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    dut.in_first.value = 0
    dut.in_last.value = 0
    await monitor

    errors = 0
    for t, (a, b) in enumerate(tiles):
        c = a.astype(np.int64) @ b.astype(np.int64)
        for s in range(N):
            got = results[t * N + s]
            want = list(c[N - 1 - s, :])
            if got != want:
                errors += 1
                dut._log.error("tile %d row %d: got %s want %s", t, N - 1 - s, got, want)
    return errors


def random_tile(rng, k_len, lo_b=-128):
    a = rng.integers(-128, 128, size=(N, k_len))
    b = rng.integers(lo_b, 128, size=(k_len, N))
    return a, b


@cocotb.test()
async def test_back_to_back_tiles(dut):
    """Random tiles streamed with no gap between them."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    rng = np.random.default_rng(1)
    tiles = [random_tile(rng, int(rng.integers(N + 1, 4 * N + 8))) for _ in range(30)]
    errors = await run_tiles(dut, tiles, [0] * len(tiles))
    assert errors == 0, "{} result vectors were wrong".format(errors)


@cocotb.test()
async def test_tiles_with_gaps(dut):
    """Random tiles with random idle cycles in between."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    rng = np.random.default_rng(2)
    tiles = [random_tile(rng, int(rng.integers(N + 1, 3 * N + 4))) for _ in range(30)]
    gaps = [int(rng.integers(0, 3 * N)) for _ in tiles]
    errors = await run_tiles(dut, tiles, gaps)
    assert errors == 0, "{} result vectors were wrong".format(errors)


@cocotb.test()
async def test_extreme_values(dut):
    """Largest magnitudes (-128 x -128 and -128 x 127) on long tiles."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    k_len = 784
    tiles = [
        (np.full((N, k_len), -128), np.full((k_len, N), -128)),
        (np.full((N, k_len), -128), np.full((k_len, N), 127)),
        (np.full((N, k_len), 127), np.full((k_len, N), 127)),
    ]
    errors = await run_tiles(dut, tiles, [0, 0, 5])
    assert errors == 0, "{} result vectors were wrong".format(errors)


@cocotb.test()
async def test_mlp_like_shapes(dut):
    """Tile lengths used by the real network (784, 128, 64) back to back."""
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset(dut)
    rng = np.random.default_rng(3)
    tiles = [random_tile(rng, k, lo_b=0) for k in [784, 784, 128, 128, 64, 64]]
    errors = await run_tiles(dut, tiles, [0, 0, 7, 0, 7, 0])
    assert errors == 0, "{} result vectors were wrong".format(errors)
