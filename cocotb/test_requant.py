"""
test_requant.py - cocotb tests for npu_requant against the golden formula
used by model/golden.py.
"""
import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import golden  # noqa: E402

LATENCY = 4


def to_signed(v, width):
    v &= (1 << width) - 1
    return v - (1 << width) if v >= 1 << (width - 1) else v


async def check_stream(dut, stimuli):
    """Drive one stimulus per cycle, compare outputs LATENCY cycles later."""
    expected = []
    got = []

    async def monitor():
        while len(got) < len(stimuli):
            await RisingEdge(dut.clk)
            await ReadOnly()
            if dut.out_valid.value == 1:
                got.append(to_signed(int(dut.y.value), 8))

    mon = cocotb.start_soon(monitor())
    for acc, bias, m, s, relu in stimuli:
        await RisingEdge(dut.clk)
        dut.in_valid.value = 1
        dut.acc.value = acc & 0xFFFFFFFF
        dut.bias.value = bias & 0xFFFFFFFF
        dut.mult.value = m
        dut.shift.value = s
        dut.relu.value = int(relu)
        expected.append(int(golden.requant(acc + bias, m, s, relu)))
    await RisingEdge(dut.clk)
    dut.in_valid.value = 0
    await mon
    errors = [(i, g, e) for i, (g, e) in enumerate(zip(got, expected)) if g != e]
    for i, g, e in errors[:10]:
        dut._log.error("stimulus %d %s: got %d want %d", i, stimuli[i], g, e)
    return len(errors)


async def setup(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut.rst_n.value = 0
    dut.in_valid.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.rst_n.value = 1


@cocotb.test()
async def test_random(dut):
    """Random accumulators, biases, multipliers and shifts (ReLU on and off)."""
    await setup(dut)
    rng = np.random.default_rng(7)
    stim = []
    for _ in range(3000):
        acc = int(rng.integers(-(1 << 24), 1 << 24))
        bias = int(rng.integers(-(1 << 16), 1 << 16))
        m = int(rng.integers(1 << 14, 1 << 15))
        s = int(rng.integers(16, 31))
        stim.append((acc, bias, m, s, bool(rng.integers(0, 2))))
    assert await check_stream(dut, stim) == 0


@cocotb.test()
async def test_model_constants(dut):
    """The exact M/S values of the trained network, around every rounding edge."""
    await setup(dut)
    layers = golden.load_quant_model()
    stim = []
    for layer in layers:
        m, s, relu = layer["M"], layer["S"], layer["relu"]
        for target in range(-130, 131):
            # accumulator values that land exactly on and next to .5 boundaries
            base = ((2 * target + 1) << (s - 1)) // m
            for delta in (-2, -1, 0, 1, 2):
                stim.append((base + delta, 0, m, s, relu))
    assert await check_stream(dut, stim) == 0


@cocotb.test()
async def test_saturation(dut):
    """Huge positive/negative values must clamp to 127 / -128 (or 0 with ReLU)."""
    await setup(dut)
    stim = []
    for acc in (2**31 - 1 - 70000, -(2**31) + 70000, 10**8, -(10**8)):
        for relu in (False, True):
            stim.append((acc, 0, 32767, 16, relu))
    assert await check_stream(dut, stim) == 0
