// =============================================================================
// npu_sva.sv
// SystemVerilog Assertions for TinyNPU, attached with `bind` from tb_npu.sv
// (no RTL changes).
//
//   npu_top_sva   AXI4-Lite and AXI4-Stream handshake rules, busy/done rules
//   npu_core_sva  controller and writer invariants inside the core
//
// Every failure increments `failures`; the testbench adds it to its result.
// =============================================================================
`timescale 1ns/1ps

module npu_top_sva #(parameter int N = 8) (
  input logic           aclk,
  input logic           aresetn,
  // AXI4-Lite
  input logic [11:0]    awaddr,  input logic awvalid, input logic awready,
  input logic [31:0]    wdata,   input logic wvalid,  input logic wready,
  input logic [1:0]     bresp,   input logic bvalid,  input logic bready,
  input logic [11:0]    araddr,  input logic arvalid, input logic arready,
  input logic [31:0]    rdata,   input logic [1:0] rresp,
  input logic           rvalid,  input logic rready,
  // AXI4-Stream
  input logic [8*N-1:0] tdata,   input logic tvalid,  input logic tready,
  input logic           tlast,
  // internal
  input logic           start,   input logic busy,    input logic done,
  input logic           irq_en,  input logic irq
);

  int failures = 0;

  function automatic void fail(input string msg);
    failures++;
    $error("%s", msg);
  endfunction

  // ---- AXI4-Lite: a VALID stays up, with stable payload, until READY ----
  a_awvalid_hold: assert property (@(posedge aclk) disable iff (!aresetn) awvalid && !awready |=> awvalid && $stable(awaddr))
    else fail("AWVALID/AWADDR changed before AWREADY");
  a_wvalid_hold:  assert property (@(posedge aclk) disable iff (!aresetn) wvalid && !wready |=> wvalid && $stable(wdata))
    else fail("WVALID/WDATA changed before WREADY");
  a_bvalid_hold:  assert property (@(posedge aclk) disable iff (!aresetn) bvalid && !bready |=> bvalid && $stable(bresp))
    else fail("BVALID/BRESP changed before BREADY");
  a_arvalid_hold: assert property (@(posedge aclk) disable iff (!aresetn) arvalid && !arready |=> arvalid && $stable(araddr))
    else fail("ARVALID/ARADDR changed before ARREADY");
  a_rvalid_hold:  assert property (@(posedge aclk) disable iff (!aresetn) rvalid && !rready |=> rvalid && $stable(rdata) && $stable(rresp))
    else fail("RVALID/RDATA changed before RREADY");
  a_bresp_legal:  assert property (@(posedge aclk) disable iff (!aresetn) bvalid |-> (bresp == 2'b00 || bresp == 2'b10))
    else fail("illegal BRESP code");
  a_rresp_legal:  assert property (@(posedge aclk) disable iff (!aresetn) rvalid |-> (rresp == 2'b00 || rresp == 2'b10))
    else fail("illegal RRESP code");

  // ---- AXI4-Stream: a beat is held until it is accepted ----
  a_tvalid_hold:  assert property (@(posedge aclk) disable iff (!aresetn) tvalid && !tready |=> tvalid && $stable(tdata) && $stable(tlast))
    else fail("TVALID/TDATA changed before TREADY");
  a_no_beat_busy: assert property (@(posedge aclk) disable iff (!aresetn) busy |-> !tready)
    else fail("image stream accepted during an inference");

  // ---- run control ----
  a_busy_done:    assert property (@(posedge aclk) disable iff (!aresetn) !(busy && done))
    else fail("BUSY and DONE both high");
  a_start_runs:   assert property (@(posedge aclk) disable iff (!aresetn) start && !busy |=> busy && !done)
    else fail("START did not begin an inference");
  a_irq:          assert property (@(posedge aclk) disable iff (!aresetn) irq == (done && irq_en))
    else fail("irq does not follow DONE && IRQ_EN");

endmodule


module npu_core_sva #(parameter int N = 8) (
  input logic       clk,
  input logic       rst_n,
  input logic [2:0] state,
  input logic       issue_valid,
  input logic [9:0] k,
  input logic [9:0] k_len,
  input logic       g_valid,
  input logic       rq_valid0,
  input logic [7:0] w5_neuron,
  input logic [7:0] p_len,
  input logic       layer_done
);

  localparam logic [2:0] S_IDLE = 3'd0;
  localparam logic [2:0] S_WAIT = 3'd2;

  int failures = 0;

  function automatic void fail(input string msg);
    failures++;
    $error("%s", msg);
  endfunction

  a_k_in_range:     assert property (@(posedge clk) disable iff (!rst_n) issue_valid |-> k < k_len)
    else fail("feeder index k out of range");
  a_tile_long:      assert property (@(posedge clk) disable iff (!rst_n) issue_valid |-> k_len > 10'(N))
    else fail("tile shorter than N+1: results would be overwritten");
  a_results_in_run: assert property (@(posedge clk) disable iff (!rst_n) g_valid |-> state != S_IDLE)
    else fail("array produced a result outside an inference");
  a_write_in_range: assert property (@(posedge clk) disable iff (!rst_n) rq_valid0 |-> w5_neuron < p_len)
    else fail("writer address beyond the layer width");
  a_layer_done:     assert property (@(posedge clk) disable iff (!rst_n) layer_done |-> state == S_WAIT)
    else fail("layer finished while still feeding");

endmodule
