// =============================================================================
// npu_rom.sv
// Read-only memory initialised from a hex file ($readmemh), with a synchronous
// read port. Holds the network's weights and biases; Vivado maps it to block
// RAM with the contents baked into the bitstream.
// =============================================================================
`timescale 1ns/1ps

module npu_rom #(
  parameter int W         = 64,
  parameter int D         = 1024,
  parameter     INIT_FILE = "weights.mem"
) (
  input  logic                 clk,
  input  logic [$clog2(D)-1:0] addr,
  output logic [W-1:0]         data
);

  (* rom_style = "block" *) logic [W-1:0] mem [0:D-1];

  initial begin
    $readmemh(INIT_FILE, mem);
  end

  always_ff @(posedge clk) begin
    data <= mem[addr];
  end

endmodule
