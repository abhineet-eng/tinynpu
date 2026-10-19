// =============================================================================
// npu_gemm.sv
// Matrix-multiply engine: input skew + N x N systolic array + output de-skew.
//
// Each cycle the controller presents one "k" slice of a tile:
//   in_a[8i+7:8i]  weight     A[i][k]  for output row i      (i = 0..N-1)
//   in_b[8j+7:8j]  activation B[k][j]  for image j            (j = 0..N-1)
// with in_first on k = 0 and in_last on k = K-1. Tiles may follow each other
// with no gap. K must be greater than N.
//
// For every tile the engine outputs N result vectors on N consecutive cycles:
//   out_res[ACC_W*j +: ACC_W] = C[r][j] = sum_k A[r][k] * B[k][j]
// for r = N-1, N-2, ..., 0 (row N-1 first). All N lanes of a vector belong to
// the same output row r, so they share one bias and one requantization.
//
// Latency: first result vector appears K + 2N cycles after the tile's first
// input slice.
// =============================================================================
`timescale 1ns/1ps

module npu_gemm #(
  parameter int N     = 8,
  parameter int ACC_W = 32
) (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               in_valid,
  input  logic               in_first,
  input  logic               in_last,
  input  logic [8*N-1:0]     in_a,
  input  logic [8*N-1:0]     in_b,
  output logic               out_valid,
  output logic [ACC_W*N-1:0] out_res
);

  logic [8*N-1:0]     a_west, b_north;
  logic [N-1:0]       v_west, f_west, l_west;
  logic [ACC_W*N-1:0] res_south;
  logic [N-1:0]       res_valid;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [N-1:0]       aligned_valid;   // identical after de-skew; lane 0 is used
  /* verilator lint_on UNUSEDSIGNAL */

  genvar i;
  generate
    for (i = 0; i < N; i++) begin : g_skew
      // row i (weights + flags) delayed by i cycles
      npu_delay #(.W(11), .D(i)) u_row_skew (
        .clk  (clk),
        .rst_n(rst_n),
        .din  ({in_valid, in_first, in_last, in_a[8*i +: 8]}),
        .dout ({v_west[i], f_west[i], l_west[i], a_west[8*i +: 8]})
      );
      // column i (activations) delayed by i cycles
      npu_delay #(.W(8), .D(i)) u_col_skew (
        .clk  (clk),
        .rst_n(rst_n),
        .din  (in_b[8*i +: 8]),
        .dout (b_north[8*i +: 8])
      );
      // column i output delayed by N-1-i cycles so all columns line up
      npu_delay #(.W(ACC_W+1), .D(N-1-i)) u_deskew (
        .clk  (clk),
        .rst_n(rst_n),
        .din  ({res_valid[i], res_south[ACC_W*i +: ACC_W]}),
        .dout ({aligned_valid[i], out_res[ACC_W*i +: ACC_W]})
      );
    end
  endgenerate

  npu_array #(.N(N), .ACC_W(ACC_W)) u_array (
    .clk      (clk),
    .rst_n    (rst_n),
    .a_west   (a_west),
    .v_west   (v_west),
    .f_west   (f_west),
    .l_west   (l_west),
    .b_north  (b_north),
    .res_south(res_south),
    .res_valid(res_valid)
  );

  // after de-skew every lane is valid in the same cycle
  assign out_valid = aligned_valid[0];

endmodule
