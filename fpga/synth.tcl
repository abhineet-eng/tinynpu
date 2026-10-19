# =============================================================================
# synth.tcl - Vivado out-of-context synthesis + implementation of TinyNPU
#
#   vivado -mode batch -source synth.tcl -tclargs <N> <part> <clock_ns>
#
# Out-of-context mode implements the accelerator as an IP block (no I/O pins),
# which is the standard way to measure an IP's resources and maximum clock.
# Run from a folder that contains weights.mem and bias.mem (synth.bat does it).
# =============================================================================

set N        [lindex $argv 0]
set part     [lindex $argv 1]
set clock_ns [lindex $argv 2]
if {$N eq ""}        { set N 8 }
if {$part eq ""}     { set part xc7a35tcpg236-1 }
if {$clock_ns eq ""} { set clock_ns 10.0 }

set root [file normalize [file join [file dirname [info script]] ..]]
file mkdir reports

puts "==== TinyNPU: N=$N  part=$part  clock target=$clock_ns ns ===="

foreach f {npu_delay npu_pe npu_array npu_gemm npu_requant npu_ram npu_rom
           npu_core npu_axil_regs npu_top} {
  read_verilog -sv [file join $root rtl $f.sv]
}
read_mem weights.mem
read_mem bias.mem

synth_design -top npu_top -part $part -mode out_of_context \
             -generic N=$N -include_dirs [file join $root rtl]

create_clock -name aclk -period $clock_ns [get_ports aclk]

opt_design
place_design
phys_opt_design
route_design

report_utilization    -file reports/utilization.rpt
report_timing_summary -file reports/timing.rpt
report_power          -file reports/power.rpt
write_checkpoint -force reports/npu_routed.dcp

# ---- short summary ----
# returns {used available} for a row of the utilization table
proc util_row {report name} {
  foreach line [split $report "\n"] {
    set cells {}
    foreach c [split $line "|"] {
      set c [string trim $c]
      if {$c ne ""} { lappend cells $c }
    }
    if {[llength $cells] >= 4 && [string trimright [lindex $cells 0] "*"] eq $name} {
      return [list [lindex $cells 1] [lindex $cells end-1]]
    }
  }
  return {? ?}
}

set util [report_utilization -return_string]
lassign [util_row $util "Slice LUTs"]      luts luts_a
lassign [util_row $util "Slice Registers"] ffs  ffs_a
lassign [util_row $util "Block RAM Tile"]  bram bram_a
lassign [util_row $util "DSPs"]            dsps dsps_a

set path [get_timing_paths -delay_type max -max_paths 1 -nworst 1]
set wns  [get_property SLACK $path]
if {$wns eq ""} {
  set fmax "?"
} else {
  set fmax [format "%.1f" [expr {1000.0 / ($clock_ns - $wns)}]]
}

set power "?"
regexp {Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)} [report_power -return_string] -> power

set lines [list \
  "TinyNPU implementation summary" \
  "  Array size            : ${N} x ${N}" \
  "  Device                : $part" \
  "  Clock target          : $clock_ns ns ([format %.0f [expr {1000.0 / $clock_ns}]] MHz)" \
  "  Worst negative slack  : $wns ns" \
  "  Maximum clock (Fmax)  : $fmax MHz" \
  "  LUTs                  : $luts / $luts_a" \
  "  Flip-flops            : $ffs / $ffs_a" \
  "  Block RAM tiles       : $bram / $bram_a" \
  "  DSP48 slices          : $dsps / $dsps_a" \
  "  Total on-chip power   : $power W (vectorless estimate at the target clock)" \
]
set fh [open reports/summary.txt w]
foreach l $lines { puts $l; puts $fh $l }
close $fh
puts "Reports written to [file normalize reports]"
