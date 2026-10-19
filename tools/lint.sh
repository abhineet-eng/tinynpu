#!/usr/bin/env bash
# lint.sh - Verilator lint with all warnings enabled, for every array size.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
for n in 1 2 4 8 16; do
  echo "Linting N=$n"
  verilator --lint-only -Wall -Irtl -GN="$n" --top-module npu_top \
    rtl/npu_delay.sv rtl/npu_pe.sv rtl/npu_array.sv rtl/npu_gemm.sv rtl/npu_requant.sv \
    rtl/npu_ram.sv rtl/npu_rom.sv rtl/npu_core.sv rtl/npu_axil_regs.sv rtl/npu_top.sv
done
echo "Lint clean."
