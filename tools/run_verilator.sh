#!/usr/bin/env bash
# =============================================================================
# run_verilator.sh - build and run the full-system testbench with Verilator
#
#   tools/run_verilator.sh [N] [NUM_IMAGES] [extra plusargs...]
#   tools/run_verilator.sh 8 10000
#   tools/run_verilator.sh 8 400 +GAPS
#
# Exits with status 0 only if the testbench prints "*** TEST PASSED ***".
# =============================================================================
set -euo pipefail

N=${1:-8}
NUM=${2:-10000}
shift $(( $# > 2 ? 2 : $# )) || true

ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD="$ROOT/build/verilator_N$N"
mkdir -p "$BUILD"
cp "$ROOT/mem/N$N/weights.mem" "$ROOT/mem/N$N/bias.mem" "$BUILD/"
cd "$BUILD"

verilator --binary --timing --assert -j 0 -Wno-fatal -Wno-lint -Wno-style \
  -GN="$N" +define+NPU_SVA --top-module tb_npu -I"$ROOT/rtl" -o Vtb \
  "$ROOT"/rtl/npu_delay.sv "$ROOT"/rtl/npu_pe.sv "$ROOT"/rtl/npu_array.sv \
  "$ROOT"/rtl/npu_gemm.sv "$ROOT"/rtl/npu_requant.sv "$ROOT"/rtl/npu_ram.sv \
  "$ROOT"/rtl/npu_rom.sv "$ROOT"/rtl/npu_core.sv "$ROOT"/rtl/npu_axil_regs.sv \
  "$ROOT"/rtl/npu_top.sv "$ROOT"/tb/npu_sva.sv "$ROOT"/tb/tb_npu.sv > build.log 2>&1 \
  || { tail -40 build.log; exit 1; }

./obj_dir/Vtb +NUM_IMAGES="$NUM" +MEM_DIR="$ROOT/mem" +verilator+error+limit+1000 "$@" | tee run.log
grep -q '\*\*\* TEST PASSED \*\*\*' run.log
