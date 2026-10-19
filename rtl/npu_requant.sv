// =============================================================================
// npu_requant.sv
// Requantization lane: turns one INT32 dot product into an INT8 activation.
//
//   y = clamp( ((acc + bias) * M + 2^(S-1)) >>> S , lo, 127 )
//   lo = 0 when ReLU is on (hidden layers), -128 otherwise (output logits)
//
// M / 2^S is the fixed-point version of the real rescale factor
// (s_weight * s_input / s_output). Rounding is round-half-up.
// Four pipeline stages, so results come out 4 cycles after in_valid. M, S and
// ReLU travel down the pipeline with their data, so they may change every cycle.
// =============================================================================
`timescale 1ns/1ps

module npu_requant #(
  parameter int ACC_W = 32
) (
  input  logic                    clk,
  input  logic                    rst_n,
  input  logic                    in_valid,
  input  logic signed [ACC_W-1:0] acc,
  input  logic signed [ACC_W-1:0] bias,
  input  logic        [15:0]      mult,     // M
  input  logic        [5:0]       shift,    // S, 1..47
  input  logic                    relu,
  output logic                    out_valid,
  output logic signed [7:0]       y
);

  localparam int PW = ACC_W + 17;   // product width

  localparam logic signed [PW-1:0] MAX_Y = 127;
  localparam logic signed [PW-1:0] MIN_Y = -128;
  localparam logic signed [PW-1:0] ZERO  = 0;
  localparam logic signed [PW-1:0] ONE   = 1;

  // stage 1
  logic                    v1, relu1;
  logic signed [ACC_W-1:0] sum1;
  logic signed [16:0]      mult1;
  logic [5:0]              shift1;
  // stage 2
  logic                    v2, relu2;
  logic signed [PW-1:0]    prod2;
  logic [5:0]              shift2;
  // stage 3
  logic                    v3, relu3;
  logic signed [PW-1:0]    shifted3;

  logic signed [PW-1:0]    round_const2;
  assign round_const2 = ONE <<< (shift2 - 6'd1);

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0; out_valid <= 1'b0;
      relu1 <= 1'b0; relu2 <= 1'b0; relu3 <= 1'b0;
      sum1 <= '0; mult1 <= '0; shift1 <= '0;
      prod2 <= '0; shift2 <= '0;
      shifted3 <= '0;
      y <= '0;
    end else begin
      // stage 1: add bias
      v1     <= in_valid;
      sum1   <= acc + bias;
      mult1  <= {1'b0, mult};          // M is unsigned: give it a 0 sign bit
      shift1 <= shift;
      relu1  <= relu;

      // stage 2: multiply by M
      v2     <= v1;
      prod2  <= sum1 * mult1;
      shift2 <= shift1;
      relu2  <= relu1;

      // stage 3: round and arithmetic shift right by S
      v3       <= v2;
      shifted3 <= (prod2 + round_const2) >>> shift2;
      relu3    <= relu2;

      // stage 4: clamp to the INT8 range (and apply ReLU)
      out_valid <= v3;
      if (shifted3 > MAX_Y)
        y <= 8'sd127;
      else if (relu3 && shifted3 < ZERO)
        y <= 8'sd0;
      else if (shifted3 < MIN_Y)
        y <= -8'sd128;
      else
        y <= shifted3[7:0];
    end
  end

endmodule
