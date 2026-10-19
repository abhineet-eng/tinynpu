#!/usr/bin/env python3
"""
sweep.py - run the system testbench for every array size and print the
scaling table (cycles per image, speed-up over a single MAC, utilization).

    python sweep.py              64 images per array size
    python sweep.py --images 160

Vivado's settings64 script must already be loaded (sweep.bat does this).
Writes sim/sweep_results.csv.
"""
import argparse
import os
import re
import subprocess
import sys

SIZES = [1, 2, 4, 8, 16]
MACS_PER_IMAGE = 784 * 128 + 128 * 64 + 64 * 10
SIM_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_RE = re.compile(r"RESULT N=(\d+) images=(\d+) correct=(\d+) logit_mismatch=(\d+) "
                       r"pred_mismatch=(\d+) cycles=(\d+) mac_cycles=(\d+) batches=(\d+)")


def run_size(n, images):
    run_bat = os.path.join(SIM_DIR, "run.bat")
    proc = subprocess.run(["cmd", "/c", run_bat, str(n), str(images)], cwd=SIM_DIR,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          universal_newlines=True, errors="replace")
    m = RESULT_RE.search(proc.stdout)
    if not m:
        print(proc.stdout[-3000:])
        return None
    keys = ["n", "images", "correct", "logit_mismatch", "pred_mismatch", "cycles", "mac_cycles", "batches"]
    r = dict(zip(keys, map(int, m.groups())))
    r["passed"] = "*** TEST PASSED ***" in proc.stdout
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=64, help="images per array size (default 64)")
    args = parser.parse_args()
    if args.images % 16:
        print("Use a multiple of 16 so every array size runs full batches.")
        return 1

    rows = []
    for n in SIZES:
        print("Running N = {:2d} ({} images) ...".format(n, args.images))
        r = run_size(n, args.images)
        if r is None:
            print("  failed, see the output above")
            return 1
        rows.append(r)

    base = rows[0]["cycles"] / rows[0]["images"]
    print("\n  N   MACs  cycles/image  speed-up  utilization  mismatches  result")
    with open(os.path.join(SIM_DIR, "sweep_results.csv"), "w") as f:
        f.write("N,macs,images,cycles_per_image,speedup,utilization_pct,mismatches,passed\n")
        for r in rows:
            cpi = r["cycles"] / r["images"]
            util = 100.0 * MACS_PER_IMAGE / (cpi * r["n"] ** 2)
            mism = r["logit_mismatch"] + r["pred_mismatch"]
            print(" {:2d} {:6d}  {:12.1f}  {:7.1f}x  {:10.1f}%  {:10d}  {}".format(
                r["n"], r["n"] ** 2, cpi, base / cpi, util, mism, "PASS" if r["passed"] else "FAIL"))
            f.write("{},{},{},{:.1f},{:.2f},{:.1f},{},{}\n".format(
                r["n"], r["n"] ** 2, r["images"], cpi, base / cpi, util, mism, r["passed"]))
    print("\nSaved sim\\sweep_results.csv")
    return 0 if all(r["passed"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
