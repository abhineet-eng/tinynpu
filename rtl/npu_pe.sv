// =============================================================================
// npu_pe.sv
// One processing element (PE) of the output-stationary systolic array.
//
//   * Weights (a) and the control flags travel west -> east.
//   * Activations (b) travel north -> south.
//   * Each PE keeps its own running sum (output stationary):
//       first element of a tile : acc = a*b
//       other elements          : acc = acc + a*b
//   * One cycle after the last element of a tile, the finished sum is copied
//     into `res`. Because of this, a new tile can start streaming in the very
//     next cycle - tiles run back to back with no bubbles.
//   * `res` is read out through a per-column drain chain (d_in -> d_out),
//     loaded in parallel (`load`) and then shifted south one PE per cycle.
//
// INT8 x INT8 multiply, ACC_W-bit signed accumulate (maps onto one DSP48).
// =============================================================================
`timescale 1ns/1ps

(* use_dsp = "yes" *)
module npu_pe #(
  parameter int ACC_W = 32
) (
  input  logic              clk,
  input  logic              rst_n,

  // west -> east: weight and tile flags
  input  logic signed [7:0] a_in,
  input  logic              v_in,     // element is valid
  input  logic              f_in,     // first element of a tile
  input  logic              l_in,     // last element of a tile
  output logic signed [7:0] a_out,
  output logic              v_out,
  output logic              f_out,
  output logic              l_out,

  // north -> south: activation
  input  logic signed [7:0] b_in,
  output logic signed [7:0] b_out,

  // north -> south: result drain chain
  input  logic              load,     // copy res into the chain (whole column)
  input  logic [ACC_W-1:0]  d_in,
  input  logic              dv_in,
  output logic [ACC_W-1:0]  d_out,
  output logic              dv_out
);

  logic signed [15:0]      prod;
  logic signed [ACC_W-1:0] acc;
  logic signed [ACC_W-1:0] acc_base;   // 0 on a tile's first element
  logic signed [ACC_W-1:0] res;

  assign prod     = a_in * b_in;
  assign acc_base = f_in ? '0 : acc;   // kept as a signed signal on purpose

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      a_out  <= '0;
      b_out  <= '0;
      v_out  <= 1'b0;
      f_out  <= 1'b0;
      l_out  <= 1'b0;
      acc    <= '0;
      res    <= '0;
      d_out  <= '0;
      dv_out <= 1'b0;
    end else begin
      // pass operands and flags on to the neighbours
      a_out <= a_in;
      b_out <= b_in;
      v_out <= v_in;
      f_out <= f_in;
      l_out <= l_in;

      // multiply-accumulate
      if (v_in)
        acc <= acc_base + ACC_W'(prod);   // prod is signed: sign-extended

      // the element accepted last cycle was the tile's last: acc is final
      if (v_out && l_out)
        res <= acc;

      // drain chain: parallel load, otherwise shift south
      if (load) begin
        d_out  <= res;
        dv_out <= 1'b1;
      end else begin
        d_out  <= d_in;
        dv_out <= dv_in;
      end
    end
  end

endmodule
