// =============================================================================
// npu_axil_regs.sv
// AXI4-Lite slave with the TinyNPU register map.
//
//   0x000  ID          RO  0x4E505531 ("NPU1")
//   0x004  CONFIG      RO  [7:0] N (array size), [15:8] layers, [31:16] input size
//   0x008  CTRL        RW  [0] START (write 1, self-clearing), [1] IRQ_EN
//   0x00C  STATUS      RO  [0] BUSY, [1] DONE, [2] IMG_READY, [3] STREAM_ERR
//   0x010  CYCLES      RO  clock cycles of the last inference
//   0x014  MAC_CYCLES  RO  cycles in which the array was fed data
//   0x018  PRED0       RO  predicted digits of images 0..7   (4 bits each)
//   0x01C  PRED1       RO  predicted digits of images 8..15  (4 bits each)
//   0x020  RUNS        RO  number of completed inferences
//   0x400  LOGITS      RO  0x400 + 64*j + 4*n : logit n of image j (sign-extended)
//
// Writes to anything other than CTRL, and reads of unmapped addresses, return
// SLVERR. Write address and write data may arrive in any order.
// =============================================================================
`timescale 1ns/1ps

module npu_axil_regs #(
  parameter int N       = 8,
  parameter int IN_DIM  = 784,
  parameter int OUT_DIM = 10
) (
  input  logic           clk,
  input  logic           rst_n,

  // AXI4-Lite slave
  input  logic [11:0]    s_awaddr,
  input  logic           s_awvalid,
  output logic           s_awready,
  input  logic [31:0]    s_wdata,
  input  logic [3:0]     s_wstrb,
  input  logic           s_wvalid,
  output logic           s_wready,
  output logic [1:0]     s_bresp,
  output logic           s_bvalid,
  input  logic           s_bready,
  input  logic [11:0]    s_araddr,
  input  logic           s_arvalid,
  output logic           s_arready,
  output logic [31:0]    s_rdata,
  output logic [1:0]     s_rresp,
  output logic           s_rvalid,
  input  logic           s_rready,

  // to / from the core
  output logic           start,        // one-cycle pulse
  output logic           irq_en,
  input  logic           busy,
  input  logic           done,
  input  logic           img_ready,
  input  logic           stream_err,
  input  logic [31:0]    cycles,
  input  logic [31:0]    mac_cycles,
  input  logic [31:0]    runs,
  input  logic [4*N-1:0] preds,
  output logic [3:0]     logit_sel,
  input  logic [8*N-1:0] logit_row
);

  localparam logic [1:0]  OKAY   = 2'b00;
  localparam logic [1:0]  SLVERR = 2'b10;
  localparam logic [31:0] ID     = 32'h4E50_5531;

  // ---------------------------------------------------------------------------
  // Write path
  // ---------------------------------------------------------------------------
  logic        aw_held, w_held;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [11:0] awaddr_q;      // word address only
  logic [31:0] wdata_q;       // only CTRL[1:0] is writable
  logic [3:0]  wstrb_q;
  /* verilator lint_on UNUSEDSIGNAL */

  assign s_awready = !aw_held && !s_bvalid;
  assign s_wready  = !w_held  && !s_bvalid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      aw_held  <= 1'b0;
      w_held   <= 1'b0;
      awaddr_q <= '0;
      wdata_q  <= '0;
      wstrb_q  <= '0;
      s_bvalid <= 1'b0;
      s_bresp  <= OKAY;
      start    <= 1'b0;
      irq_en   <= 1'b0;
    end else begin
      start <= 1'b0;

      if (s_awvalid && s_awready) begin
        awaddr_q <= s_awaddr;
        aw_held  <= 1'b1;
      end
      if (s_wvalid && s_wready) begin
        wdata_q <= s_wdata;
        wstrb_q <= s_wstrb;
        w_held  <= 1'b1;
      end

      if (aw_held && w_held && !s_bvalid) begin
        if (awaddr_q[11:2] == 10'h002) begin          // CTRL
          if (wstrb_q[0]) begin
            start  <= wdata_q[0];
            irq_en <= wdata_q[1];
          end
          s_bresp <= OKAY;
        end else begin
          s_bresp <= SLVERR;
        end
        s_bvalid <= 1'b1;
        aw_held  <= 1'b0;
        w_held   <= 1'b0;
      end

      if (s_bvalid && s_bready)
        s_bvalid <= 1'b0;
    end
  end

  // ---------------------------------------------------------------------------
  // Read path
  // ---------------------------------------------------------------------------
  logic [31:0] pred_lo, pred_hi;
  logic [3:0]  logit_img;
  logic        logit_hit;

  // predictions of images 0..7 in PRED0, 8..15 in PRED1 (unused nibbles are 0)
  logic [63:0] preds_64;
  assign preds_64 = 64'(preds);
  assign pred_lo  = preds_64[31:0];
  assign pred_hi  = preds_64[63:32];

  // logits window: 0x400 + 64*image + 4*logit
  assign logit_sel = s_araddr[5:2];
  assign logit_img = s_araddr[9:6];
  assign logit_hit = (s_araddr[11:10] == 2'b01) &&
                     (32'(logit_img) < N) && (32'(logit_sel) < OUT_DIM);

  assign s_arready = !s_rvalid;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s_rvalid <= 1'b0;
      s_rdata  <= '0;
      s_rresp  <= OKAY;
    end else begin
      if (s_arvalid && s_arready) begin
        s_rresp <= OKAY;
        if (logit_hit) begin
          s_rdata <= {{24{logit_row[8*logit_img + 7]}}, logit_row[8*logit_img +: 8]};
        end else begin
          case (s_araddr)
            12'h000: s_rdata <= ID;
            12'h004: s_rdata <= {16'(IN_DIM), 8'd3, 8'(N)};
            12'h008: s_rdata <= {30'd0, irq_en, 1'b0};
            12'h00C: s_rdata <= {28'd0, stream_err, img_ready, done, busy};
            12'h010: s_rdata <= cycles;
            12'h014: s_rdata <= mac_cycles;
            12'h018: s_rdata <= pred_lo;
            12'h01C: s_rdata <= pred_hi;
            12'h020: s_rdata <= runs;
            default: begin
              s_rdata <= '0;
              s_rresp <= SLVERR;
            end
          endcase
        end
        s_rvalid <= 1'b1;
      end else if (s_rvalid && s_rready) begin
        s_rvalid <= 1'b0;
      end
    end
  end

endmodule
