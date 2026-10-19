// =============================================================================
// npu_delay.sv
// Fixed-length delay line (shift register) of D clock cycles, W bits wide.
// D = 0 is a plain wire. Used to skew the systolic-array inputs and to
// de-skew its outputs.
// =============================================================================
`timescale 1ns/1ps

module npu_delay #(
  parameter int W = 8,
  parameter int D = 1
) (
  /* verilator lint_off UNUSEDSIGNAL */
  input  logic         clk,      // unused when D = 0
  input  logic         rst_n,
  /* verilator lint_on UNUSEDSIGNAL */
  input  logic [W-1:0] din,
  output logic [W-1:0] dout
);

  generate
    if (D == 0) begin : g_wire
      assign dout = din;
    end else begin : g_pipe
      logic [W-1:0] pipe [0:D-1];
      always_ff @(posedge clk) begin
        if (!rst_n) begin
          for (int s = 0; s < D; s++) pipe[s] <= '0;
        end else begin
          pipe[0] <= din;
          for (int s = 1; s < D; s++) pipe[s] <= pipe[s-1];
        end
      end
      assign dout = pipe[D-1];
    end
  endgenerate

endmodule
