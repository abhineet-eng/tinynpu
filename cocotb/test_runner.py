"""
test_runner.py - builds each RTL unit with Icarus Verilog and runs its cocotb
tests. Run all of them with:

    cd cocotb
    pytest -v test_runner.py

Requirements: Icarus Verilog 12+, Python packages cocotb (1.9.x), numpy, pytest.
"""
import os
import shutil

import pytest
from cocotb.runner import get_runner

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RTL = os.path.join(ROOT, "rtl")
SIM = os.environ.get("SIM", "icarus")


def rtl(*names):
    return [os.path.join(RTL, n) for n in names]


def run(toplevel, sources, test_module, parameters, build_name, extra_env=None, files=()):
    build_dir = os.path.join(HERE, "sim_build", build_name)
    os.makedirs(build_dir, exist_ok=True)
    for src, dst in files:
        shutil.copy(src, os.path.join(build_dir, dst))
    runner = get_runner(SIM)
    runner.build(
        sources=sources,
        hdl_toplevel=toplevel,
        parameters=parameters,
        includes=[RTL],
        build_dir=build_dir,
        always=True,
        timescale=("1ns", "1ps"),
    )
    env = {k: str(v) for k, v in parameters.items()}
    env.update(extra_env or {})
    runner.test(
        hdl_toplevel=toplevel,
        test_module=test_module,
        build_dir=build_dir,
        extra_env=env,
    )


@pytest.mark.parametrize("n", [2, 4, 8])
def test_gemm(n):
    run("npu_gemm", rtl("npu_delay.sv", "npu_pe.sv", "npu_array.sv", "npu_gemm.sv"),
        "test_gemm", {"N": n}, "gemm_N{}".format(n))


def test_requant():
    run("npu_requant", rtl("npu_requant.sv"), "test_requant", {}, "requant")


def test_top():
    """Full accelerator on real MNIST images (8 x 8 array, 2 batches)."""
    n = 8
    mem = os.path.join(ROOT, "mem", "N{}".format(n))
    run("npu_top",
        rtl("npu_delay.sv", "npu_pe.sv", "npu_array.sv", "npu_gemm.sv", "npu_requant.sv",
            "npu_ram.sv", "npu_rom.sv", "npu_core.sv", "npu_axil_regs.sv", "npu_top.sv"),
        "test_top", {"N": n}, "top_N{}".format(n),
        extra_env={"BATCHES": "2"},
        files=[(os.path.join(mem, "weights.mem"), "weights.mem"),
               (os.path.join(mem, "bias.mem"), "bias.mem")])
