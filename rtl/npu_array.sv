// =============================================================================
// npu_array.sv
// N x N grid of processing elements (output-stationary systolic array).
//
// Inputs must already be skewed: row i is delayed by i cycles and column j by
// j cycles (see npu_gemm.sv), so that weight a[i][k] and activation b[k][j]
// meet in PE(i,j) in the same cycle.
//
// PE(i,j) computes C[i][j] = sum_k A[i][k] * B[k][j] for one tile.
//
// Result drain: when the bottom PE of column j has finished a tile, the whole
// column copies its results into the drain chain one cycle later, then shifts
// them out of the bottom: row N-1 first, then N-2, ..., row 0 (one per cycle).
// Column j's output stream is therefore j cycles later than column 0's.
//
// Requirement: every tile is at least N+1 elements long (K > N), so a column's
// results are read out before the next tile overwrites them.
//
// Wiring between neighbours uses 2-D arrays of nets:
//   *_h[i][j]  horizontal link into PE(i,j) from the west (j = N: east edge)
//   *_v[i][j]  vertical   link into PE(i,j) from the north (i = N: south edge)
// =============================================================================
`timescale 1ns/1ps

module npu_array #(
  parameter int N     = 8,
  parameter int ACC_W = 32
) (
  input  logic               clk,
  input  logic               rst_n,

  // west edge (skewed), one lane per row
  input  logic [8*N-1:0]     a_west,
  input  logic [N-1:0]       v_west,
  input  logic [N-1:0]       f_west,
  input  logic [N-1:0]       l_west,

  // north edge (skewed), one lane per column
  input  logic [8*N-1:0]     b_north,

  // south edge: drained results, one lane per column
  output logic [ACC_W*N-1:0] res_south,
  output logic [N-1:0]       res_valid
);

  wire [7:0]       a_h  [0:N-1][0:N];
  wire             v_h  [0:N-1][0:N];
  wire             f_h  [0:N-1][0:N];
  wire             l_h  [0:N-1][0:N];
  wire [7:0]       b_v  [0:N][0:N-1];
  wire [ACC_W-1:0] d_v  [0:N][0:N-1];
  wire             dv_v [0:N][0:N-1];
  logic [N-1:0]    load_col;

  genvar i, j;
  generate
    // west and north edges
    for (i = 0; i < N; i++) begin : g_west
      assign a_h[i][0] = a_west[8*i +: 8];
      assign v_h[i][0] = v_west[i];
      assign f_h[i][0] = f_west[i];
      assign l_h[i][0] = l_west[i];
    end
    for (j = 0; j < N; j++) begin : g_north
      assign b_v[0][j]  = b_north[8*j +: 8];
      assign d_v[0][j]  = '0;        // top of every drain chain
      assign dv_v[0][j] = 1'b0;
    end

    // the PE grid
    for (i = 0; i < N; i++) begin : g_row
      for (j = 0; j < N; j++) begin : g_col
        npu_pe #(.ACC_W(ACC_W)) u_pe (
          .clk    (clk),
          .rst_n  (rst_n),
          .a_in   (a_h[i][j]),
          .v_in   (v_h[i][j]),
          .f_in   (f_h[i][j]),
          .l_in   (l_h[i][j]),
          .a_out  (a_h[i][j+1]),
          .v_out  (v_h[i][j+1]),
          .f_out  (f_h[i][j+1]),
          .l_out  (l_h[i][j+1]),
          .b_in   (b_v[i][j]),
          .b_out  (b_v[i+1][j]),
          .load   (load_col[j]),
          .d_in   (d_v[i][j]),
          .dv_in  (dv_v[i][j]),
          .d_out  (d_v[i+1][j]),
          .dv_out (dv_v[i+1][j])
        );
      end
    end

    // Column j drains one cycle after its bottom PE (row N-1) has latched the
    // finished sum, i.e. two cycles after that PE saw the tile's last element.
    for (j = 0; j < N; j++) begin : g_load
      always_ff @(posedge clk) begin
        if (!rst_n) load_col[j] <= 1'b0;
        else        load_col[j] <= v_h[N-1][j+1] & l_h[N-1][j+1];
      end
      assign res_south[ACC_W*j +: ACC_W] = d_v[N][j];
      assign res_valid[j]                = dv_v[N][j];
    end
  endgenerate

endmodule
