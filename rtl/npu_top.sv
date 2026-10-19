// =============================================================================
// npu_top.sv
// TinyNPU: an INT8 neural-network accelerator for a 784-128-64-10 MLP.
//
// Interfaces
//   AXI4-Lite slave   control and status registers (see npu_axil_regs.sv)
//   AXI4-Stream slave image input. One beat per pixel position k = 0..783;
//                     byte j of the beat is pixel k of image j (j = 0..N-1),
//                     raw 8-bit grey levels. TLAST on the last beat.
//   irq               high while DONE and IRQ_EN are both set
//
// Use
//   1. stream 784 beats (one batch of N images)
//   2. write CTRL = 1 (START), or 3 to also enable the interrupt
//   3. wait for STATUS.DONE (or irq), read PRED0/PRED1 and the counters
// TREADY is low while an inference is running.
// =============================================================================
`timescale 1ns/1ps

module npu_top #(
  parameter int N           = 8,
  parameter     WEIGHT_FILE = "weights.mem",
  parameter     BIAS_FILE   = "bias.mem"
) (
  input  logic           aclk,
  input  logic           aresetn,

  // AXI4-Lite control
  input  logic [11:0]    s_axil_awaddr,
  input  logic           s_axil_awvalid,
  output logic           s_axil_awready,
  input  logic [31:0]    s_axil_wdata,
  input  logic [3:0]     s_axil_wstrb,
  input  logic           s_axil_wvalid,
  output logic           s_axil_wready,
  output logic [1:0]     s_axil_bresp,
  output logic           s_axil_bvalid,
  input  logic           s_axil_bready,
  input  logic [11:0]    s_axil_araddr,
  input  logic           s_axil_arvalid,
  output logic           s_axil_arready,
  output logic [31:0]    s_axil_rdata,
  output logic [1:0]     s_axil_rresp,
  output logic           s_axil_rvalid,
  input  logic           s_axil_rready,

  // AXI4-Stream image input (bit 0 of each pixel is dropped by the >>1 conversion)
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic [8*N-1:0] s_axis_tdata,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic           s_axis_tvalid,
  output logic           s_axis_tready,
  input  logic           s_axis_tlast,

  output logic           irq
);

  localparam int IN_DIM  = 784;
  localparam int OUT_DIM = 10;

  logic           start, irq_en, busy, done;
  logic [31:0]    cycles, mac_cycles, runs;
  logic [4*N-1:0] preds;
  logic [3:0]     logit_sel;
  logic [8*N-1:0] logit_row;

  // ---------------------------------------------------------------------------
  // AXI4-Stream image input
  // ---------------------------------------------------------------------------
  logic           img_we;
  logic [9:0]     img_waddr;
  logic [8*N-1:0] img_wdata;
  logic [9:0]     wptr;
  logic           img_ready, stream_err;
  logic           beat;

  assign s_axis_tready = !busy;
  assign beat          = s_axis_tvalid && s_axis_tready;

  // pixel 0..255 -> INT8 activation 0..127
  genvar j;
  generate
    for (j = 0; j < N; j++) begin : g_px
      assign img_wdata[8*j +: 8] = {1'b0, s_axis_tdata[8*j+1 +: 7]};
    end
  endgenerate

  assign img_we    = beat;
  assign img_waddr = wptr;

  always_ff @(posedge aclk) begin
    if (!aresetn) begin
      wptr       <= '0;
      img_ready  <= 1'b0;
      stream_err <= 1'b0;
    end else begin
      if (start && !busy) begin
        img_ready  <= 1'b0;
        stream_err <= 1'b0;
      end
      if (beat) begin
        if (wptr == 10'(IN_DIM - 1) || s_axis_tlast) begin
          wptr <= '0;
          if (wptr == 10'(IN_DIM - 1)) img_ready <= 1'b1;
          // TLAST must come exactly on the last pixel
          if ((wptr == 10'(IN_DIM - 1)) != s_axis_tlast) stream_err <= 1'b1;
        end else begin
          wptr <= wptr + 10'd1;
        end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Registers and core
  // ---------------------------------------------------------------------------
  npu_axil_regs #(.N(N), .IN_DIM(IN_DIM), .OUT_DIM(OUT_DIM)) u_regs (
    .clk        (aclk),
    .rst_n      (aresetn),
    .s_awaddr   (s_axil_awaddr),
    .s_awvalid  (s_axil_awvalid),
    .s_awready  (s_axil_awready),
    .s_wdata    (s_axil_wdata),
    .s_wstrb    (s_axil_wstrb),
    .s_wvalid   (s_axil_wvalid),
    .s_wready   (s_axil_wready),
    .s_bresp    (s_axil_bresp),
    .s_bvalid   (s_axil_bvalid),
    .s_bready   (s_axil_bready),
    .s_araddr   (s_axil_araddr),
    .s_arvalid  (s_axil_arvalid),
    .s_arready  (s_axil_arready),
    .s_rdata    (s_axil_rdata),
    .s_rresp    (s_axil_rresp),
    .s_rvalid   (s_axil_rvalid),
    .s_rready   (s_axil_rready),
    .start      (start),
    .irq_en     (irq_en),
    .busy       (busy),
    .done       (done),
    .img_ready  (img_ready),
    .stream_err (stream_err),
    .cycles     (cycles),
    .mac_cycles (mac_cycles),
    .runs       (runs),
    .preds      (preds),
    .logit_sel  (logit_sel),
    .logit_row  (logit_row)
  );

  npu_core #(.N(N), .WEIGHT_FILE(WEIGHT_FILE), .BIAS_FILE(BIAS_FILE)) u_core (
    .clk        (aclk),
    .rst_n      (aresetn),
    .start      (start),
    .busy       (busy),
    .done       (done),
    .cycles     (cycles),
    .mac_cycles (mac_cycles),
    .runs       (runs),
    .img_we     (img_we),
    .img_waddr  (img_waddr),
    .img_wdata  (img_wdata),
    .preds      (preds),
    .logit_sel  (logit_sel),
    .logit_row  (logit_row)
  );

  assign irq = done && irq_en;

endmodule
