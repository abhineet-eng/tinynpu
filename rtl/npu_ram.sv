// =============================================================================
// npu_ram.sv
// Simple dual-port RAM: one synchronous write port, one synchronous read port
// (read data appears one cycle after the address). Vivado maps it to block RAM
// or distributed RAM depending on its size.
// =============================================================================
`timescale 1ns/1ps

module npu_ram #(
  parameter int W = 64,
  parameter int D = 1024
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [$clog2(D)-1:0] waddr,
  input  logic [W-1:0]         wdata,
  input  logic [$clog2(D)-1:0] raddr,
  output logic [W-1:0]         rdata
);

  logic [W-1:0] mem [0:D-1];

  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata <= mem[raddr];
  end

endmodule
