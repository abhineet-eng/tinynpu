"""
test_top.py - cocotb test of the complete accelerator (npu_top).

Real MNIST test images are pushed through the AXI4-Stream port, the run is
started over AXI4-Lite, and every logit and prediction is compared with the
bit-exact Python model (model/golden.py).
"""
import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ReadOnly, RisingEdge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "model"))
import golden  # noqa: E402
import mnist   # noqa: E402

N = int(os.environ.get("N", "8"))
BATCHES = int(os.environ.get("BATCHES", "2"))
IN_DIM = 784
OUT_DIM = 10


class AxiLiteMaster:
    def __init__(self, dut):
        self.dut = dut
        dut.s_axil_awvalid.value = 0
        dut.s_axil_wvalid.value = 0
        dut.s_axil_arvalid.value = 0
        dut.s_axil_bready.value = 1
        dut.s_axil_rready.value = 1
        dut.s_axil_wstrb.value = 0xF

    async def write(self, addr, data):
        d = self.dut
        await RisingEdge(d.aclk)
        d.s_axil_awaddr.value = addr
        d.s_axil_awvalid.value = 1
        d.s_axil_wdata.value = data
        d.s_axil_wvalid.value = 1
        aw_done = w_done = False
        while not (aw_done and w_done):
            await RisingEdge(d.aclk)
            if d.s_axil_awvalid.value and d.s_axil_awready.value:
                aw_done = True
                d.s_axil_awvalid.value = 0
            if d.s_axil_wvalid.value and d.s_axil_wready.value:
                w_done = True
                d.s_axil_wvalid.value = 0
        while True:
            await RisingEdge(d.aclk)
            if d.s_axil_bvalid.value:
                return int(d.s_axil_bresp.value)

    async def read(self, addr):
        d = self.dut
        await RisingEdge(d.aclk)
        d.s_axil_araddr.value = addr
        d.s_axil_arvalid.value = 1
        while True:
            await RisingEdge(d.aclk)
            if d.s_axil_arready.value:
                break
        d.s_axil_arvalid.value = 0
        while True:
            await ReadOnly()
            if d.s_axil_rvalid.value:
                data, resp = int(d.s_axil_rdata.value), int(d.s_axil_rresp.value)
                await RisingEdge(d.aclk)
                return data, resp
            await RisingEdge(d.aclk)


async def stream_images(dut, images):
    """images: uint8 [N, 784]; beat k carries pixel k of every image."""
    await RisingEdge(dut.aclk)
    for k in range(IN_DIM):
        word = 0
        for j in range(N):
            word |= int(images[j, k]) << (8 * j)
        dut.s_axis_tdata.value = word
        dut.s_axis_tvalid.value = 1
        dut.s_axis_tlast.value = int(k == IN_DIM - 1)
        # the beat is taken on the first edge where TREADY was high
        while True:
            await RisingEdge(dut.aclk)
            if dut.s_axis_tready.value:
                break
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value = 0


def to_signed8(v):
    v &= 0xFF
    return v - 256 if v >= 128 else v


@cocotb.test()
async def test_mnist_batches(dut):
    """Run real MNIST batches and compare every logit with the golden model."""
    cocotb.start_soon(Clock(dut.aclk, 10, units="ns").start())
    dut.aresetn.value = 0
    dut.s_axis_tvalid.value = 0
    dut.s_axis_tlast.value = 0
    dut.s_axis_tdata.value = 0
    axil = AxiLiteMaster(dut)
    for _ in range(5):
        await RisingEdge(dut.aclk)
    dut.aresetn.value = 1

    assert (await axil.read(0x000)) == (0x4E505531, 0), "ID register"
    cfg, _ = await axil.read(0x004)
    assert cfg & 0xFF == N, "CONFIG.N"
    assert (await axil.write(0x010, 1)) == 2, "write to a read-only register must give SLVERR"

    layers = golden.load_quant_model()
    vx, vy = mnist.load_test()
    total_cycles = 0
    for b in range(BATCHES):
        imgs = vx[b * N:(b + 1) * N]
        want_logits, want_preds = golden.infer(layers, imgs)

        await stream_images(dut, imgs)
        status, _ = await axil.read(0x00C)
        assert status & 0x4, "IMG_READY not set after 784 beats"
        assert await axil.write(0x008, 0x3) == 0          # START + IRQ_EN

        # wait for the previous DONE to clear, then for the new interrupt
        for _ in range(10):
            await RisingEdge(dut.aclk)
            if not dut.irq.value:
                break
        while not dut.irq.value:
            await RisingEdge(dut.aclk)

        cycles, _ = await axil.read(0x010)
        total_cycles += cycles
        pred0, _ = await axil.read(0x018)
        pred1, _ = await axil.read(0x01C)
        preds = (pred1 << 32) | pred0
        for j in range(N):
            got_pred = (preds >> (4 * j)) & 0xF
            got_logits = []
            for n in range(OUT_DIM):
                v, resp = await axil.read(0x400 + 64 * j + 4 * n)
                assert resp == 0
                got_logits.append(to_signed8(v))
            assert got_logits == list(want_logits[j]), \
                "image {}: logits {} != golden {}".format(b * N + j, got_logits, list(want_logits[j]))
            assert got_pred == want_preds[j], "image {}: prediction".format(b * N + j)
        dut._log.info("batch %d: %d cycles, predictions %s, labels %s",
                      b, cycles, list(want_preds), list(vy[b * N:(b + 1) * N]))
    dut._log.info("average %.1f cycles per image", total_cycles / (BATCHES * N))
