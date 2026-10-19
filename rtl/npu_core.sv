// =============================================================================
// npu_core.sv
// TinyNPU compute core: runs the whole 3-layer MLP on a batch of N images.
//
//   image buffer --+
//   H1 buffer -----+--> activations (columns) --+
//   H2 buffer -----+                            +--> npu_gemm --> bias + requant
//   weight ROM -------> weights     (rows)    --+       (N x N)     + ReLU (N lanes)
//                                                                    |
//                         H1 / H2 / logits buffers  <----------------+
//                         logits --> argmax --> predictions
//
// Scheduling. A layer with K inputs and T*N outputs is processed as T tiles.
// Tile t computes outputs t*N .. t*N+N-1 for all N images at once:
//   for k = 0..K-1 : row i gets W[t*N+i][k], column j gets x_j[k]
// Tiles of a layer are streamed back to back, so the array is busy every
// cycle while a layer is running. The next layer starts once every output of
// the current layer has been written.
//
// The weight ROM holds all layers' tiles in the order they are used, so its
// address simply counts up from 0 during an inference.
// =============================================================================
`timescale 1ns/1ps

module npu_core #(
  parameter int N           = 8,
  parameter     WEIGHT_FILE = "weights.mem",
  parameter     BIAS_FILE   = "bias.mem"
) (
  input  logic                 clk,
  input  logic                 rst_n,

  // control / status
  input  logic                 start,
  output logic                 busy,
  output logic                 done,         // stays high until the next start
  output logic [31:0]          cycles,       // clock cycles of the last run
  output logic [31:0]          mac_cycles,   // cycles the array was fed data
  output logic [31:0]          runs,         // completed inferences

  // image buffer write port (driven by the AXI4-Stream input)
  input  logic                 img_we,
  input  logic [9:0]           img_waddr,
  input  logic [8*N-1:0]       img_wdata,

  // results
  output logic [4*N-1:0]       preds,        // predicted digit of image j in [4j+3:4j]
  input  logic [3:0]           logit_sel,    // debug read of the logits buffer
  output logic [8*N-1:0]       logit_row     // logit 'logit_sel' of every image
);

  `include "npu_model_params.svh"

  // ---------------------------------------------------------------------------
  // Sizes
  // ---------------------------------------------------------------------------
  localparam int ACC_W  = 32;
  localparam int T1     = (H1_DIM  + N - 1) / N;      // tiles per layer
  localparam int T2     = (H2_DIM  + N - 1) / N;
  localparam int T3     = (OUT_DIM + N - 1) / N;
  localparam int P1     = T1 * N;                      // padded layer widths
  localparam int P2     = T2 * N;
  localparam int P3     = T3 * N;
  localparam int WDEPTH = T1 * IN_DIM + T2 * H1_DIM + T3 * H2_DIM;
  localparam int BDEPTH = P1 + P2 + P3;
  localparam int WA_W   = $clog2(WDEPTH);
  localparam int BA_W   = $clog2(BDEPTH);

  localparam logic [2:0] S_IDLE   = 3'd0;   // waiting for start
  localparam logic [2:0] S_FEED   = 3'd1;   // streaming a layer's tiles
  localparam logic [2:0] S_WAIT   = 3'd2;   // waiting for the layer's last write
  localparam logic [2:0] S_ARGMAX = 3'd3;   // picking the largest logit
  localparam logic [2:0] S_DONE   = 3'd4;   // publishing the results

  // ---------------------------------------------------------------------------
  // Controller: feeds tiles into the systolic array
  // ---------------------------------------------------------------------------
  logic [2:0]      state;
  logic [1:0]      layer;            // 0, 1, 2
  logic [9:0]      k;                // input index within the layer
  logic [7:0]      tile;             // tile index within the layer
  logic [WA_W-1:0] waddr;            // weight ROM address
  logic            layer_done;       // pulse from the writer

  logic [9:0]      k_len;            // K of the current layer
  logic [7:0]      t_len;            // number of tiles of the current layer
  always_comb begin
    case (layer)
      2'd0:    begin k_len = 10'(IN_DIM); t_len = 8'(T1); end
      2'd1:    begin k_len = 10'(H1_DIM); t_len = 8'(T2); end
      default: begin k_len = 10'(H2_DIM); t_len = 8'(T3); end
    endcase
  end

  logic issue_valid, issue_first, issue_last;
  assign issue_valid = (state == S_FEED);
  assign issue_first = (k == 10'd0);
  assign issue_last  = (k == k_len - 10'd1);

  logic [3:0]      am_n;             // argmax: logit index being compared

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      state      <= S_IDLE;
      layer      <= 2'd0;
      k          <= '0;
      tile       <= '0;
      waddr      <= '0;
      busy       <= 1'b0;
      am_n       <= '0;
    end else begin
      case (state)
        S_IDLE: begin
          if (start) begin
            state <= S_FEED;
            busy  <= 1'b1;
            layer <= 2'd0;
            k     <= '0;
            tile  <= '0;
            waddr <= '0;
          end
        end

        S_FEED: begin
          waddr <= waddr + 1'b1;
          if (issue_last) begin
            k <= '0;
            if (tile == t_len - 8'd1) state <= S_WAIT;
            else                       tile  <= tile + 8'd1;
          end else begin
            k <= k + 10'd1;
          end
        end

        S_WAIT: begin
          if (layer_done) begin
            if (layer == 2'd2) begin
              state <= S_ARGMAX;
              am_n  <= '0;
            end else begin
              state <= S_FEED;
              layer <= layer + 2'd1;
              tile  <= '0;
              k     <= '0;
            end
          end
        end

        S_ARGMAX: begin
          if (am_n == 4'(OUT_DIM - 1)) state <= S_DONE;
          else                         am_n  <= am_n + 4'd1;
        end

        S_DONE: begin
          state <= S_IDLE;
          busy  <= 1'b0;
        end

        default: state <= S_IDLE;
      endcase
    end
  end

  // ---------------------------------------------------------------------------
  // Memories (all reads take one cycle)
  // ---------------------------------------------------------------------------
  logic [8*N-1:0] w_q, img_q, h1_q, h2_q;
  logic [8*N-1:0] h_wdata;
  logic           h1_we, h2_we;
  /* verilator lint_off UNUSEDSIGNAL */
  logic [7:0]     h_waddr;     // upper bits unused for small buffers
  /* verilator lint_on UNUSEDSIGNAL */

  npu_rom #(.W(8*N), .D(WDEPTH), .INIT_FILE(WEIGHT_FILE)) u_weights (
    .clk (clk),
    .addr(waddr),
    .data(w_q)
  );

  npu_ram #(.W(8*N), .D(IN_DIM)) u_img_buf (
    .clk  (clk),
    .we   (img_we),
    .waddr(img_waddr[$clog2(IN_DIM)-1:0]),
    .wdata(img_wdata),
    .raddr(k[$clog2(IN_DIM)-1:0]),
    .rdata(img_q)
  );

  npu_ram #(.W(8*N), .D(P1)) u_h1_buf (
    .clk  (clk),
    .we   (h1_we),
    .waddr(h_waddr[$clog2(P1)-1:0]),
    .wdata(h_wdata),
    .raddr(k[$clog2(P1)-1:0]),
    .rdata(h1_q)
  );

  npu_ram #(.W(8*N), .D(P2)) u_h2_buf (
    .clk  (clk),
    .we   (h2_we),
    .waddr(h_waddr[$clog2(P2)-1:0]),
    .wdata(h_wdata),
    .raddr(k[$clog2(P2)-1:0]),
    .rdata(h2_q)
  );

  // ---------------------------------------------------------------------------
  // Feed the systolic array (flags delayed one cycle to match the memories)
  // ---------------------------------------------------------------------------
  logic           s1_valid, s1_first, s1_last;
  logic [8*N-1:0] s1_act;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      s1_valid <= 1'b0;
      s1_first <= 1'b0;
      s1_last  <= 1'b0;
    end else begin
      s1_valid <= issue_valid;
      s1_first <= issue_first;
      s1_last  <= issue_last;
    end
  end

  always_comb begin
    case (layer)
      2'd0:    s1_act = img_q;
      2'd1:    s1_act = h1_q;
      default: s1_act = h2_q;
    endcase
  end

  logic               g_valid;
  logic [ACC_W*N-1:0] g_res;

  npu_gemm #(.N(N), .ACC_W(ACC_W)) u_gemm (
    .clk      (clk),
    .rst_n    (rst_n),
    .in_valid (s1_valid),
    .in_first (s1_first),
    .in_last  (s1_last),
    .in_a     (w_q),
    .in_b     (s1_act),
    .out_valid(g_valid),
    .out_res  (g_res)
  );

  // ---------------------------------------------------------------------------
  // Writer: bias + requantization + ReLU, then store the next layer's input.
  // The array returns rows N-1 .. 0 of each tile, one row per cycle.
  // ---------------------------------------------------------------------------
  logic [$clog2(N+1)-1:0] wr_step;
  logic [7:0]             wr_tile;
  logic [7:0]             w0_neuron;
  logic [BA_W-1:0]        bias_addr;
  logic [31:0]            bias_q;

  assign w0_neuron = 8'(int'(wr_tile) * N + (N - 1 - int'(wr_step)));

  always_comb begin
    case (layer)
      2'd0:    bias_addr = BA_W'(w0_neuron);
      2'd1:    bias_addr = BA_W'(P1 + int'(w0_neuron));
      default: bias_addr = BA_W'(P1 + P2 + int'(w0_neuron));
    endcase
  end

  npu_rom #(.W(32), .D(BDEPTH), .INIT_FILE(BIAS_FILE)) u_bias (
    .clk (clk),
    .addr(bias_addr),
    .data(bias_q)
  );

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wr_step <= '0;
      wr_tile <= '0;
    end else if (g_valid) begin
      if (int'(wr_step) == N - 1) begin
        wr_step <= '0;
        wr_tile <= (wr_tile == t_len - 8'd1) ? 8'd0 : wr_tile + 8'd1;
      end else begin
        wr_step <= wr_step + 1'b1;
      end
    end
  end

  // stage w1: bias is available, hand everything to the requant lanes
  logic               w1_valid;
  logic [ACC_W*N-1:0] w1_res;
  logic [7:0]         w1_neuron;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      w1_valid  <= 1'b0;
      w1_res    <= '0;
      w1_neuron <= '0;
    end else begin
      w1_valid  <= g_valid;
      w1_res    <= g_res;
      w1_neuron <= w0_neuron;
    end
  end

  logic [15:0] rq_mult;
  logic [5:0]  rq_shift;
  logic        rq_relu;
  always_comb begin
    case (layer)
      2'd0:    begin rq_mult = 16'(L1_M); rq_shift = 6'(L1_S); rq_relu = 1'b1; end
      2'd1:    begin rq_mult = 16'(L2_M); rq_shift = 6'(L2_S); rq_relu = 1'b1; end
      default: begin rq_mult = 16'(L3_M); rq_shift = 6'(L3_S); rq_relu = 1'b0; end
    endcase
  end

  /* verilator lint_off UNUSEDSIGNAL */
  logic [N-1:0] rq_valid;      // all lanes are identical; lane 0 is used
  /* verilator lint_on UNUSEDSIGNAL */
  logic [8*N-1:0] rq_y;

  genvar j;
  generate
    for (j = 0; j < N; j++) begin : g_requant
      npu_requant #(.ACC_W(ACC_W)) u_rq (
        .clk      (clk),
        .rst_n    (rst_n),
        .in_valid (w1_valid),
        .acc      (w1_res[ACC_W*j +: ACC_W]),
        .bias     (bias_q),
        .mult     (rq_mult),
        .shift    (rq_shift),
        .relu     (rq_relu),
        .out_valid(rq_valid[j]),
        .y        (rq_y[8*j +: 8])
      );
    end
  endgenerate

  // the neuron index follows its data through the 4 requant stages
  logic [7:0] w5_neuron;
  npu_delay #(.W(8), .D(4)) u_neuron_delay (
    .clk  (clk),
    .rst_n(rst_n),
    .din  (w1_neuron),
    .dout (w5_neuron)
  );

  logic [8*N-1:0] logits [0:P3-1];
  logic [7:0]     wr_count;

  assign h_wdata = rq_y;
  assign h_waddr = w5_neuron;
  assign h1_we   = rq_valid[0] && (layer == 2'd0);
  assign h2_we   = rq_valid[0] && (layer == 2'd1);

  always_ff @(posedge clk) begin
    if (rq_valid[0] && layer == 2'd2)
      logits[w5_neuron[$clog2(P3)-1:0]] <= rq_y;
  end

  logic [7:0] p_len;   // padded outputs of the current layer
  always_comb begin
    case (layer)
      2'd0:    p_len = 8'(P1);
      2'd1:    p_len = 8'(P2);
      default: p_len = 8'(P3);
    endcase
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      wr_count   <= '0;
      layer_done <= 1'b0;
    end else begin
      layer_done <= 1'b0;
      if (rq_valid[0]) begin
        if (wr_count == p_len - 8'd1) begin
          wr_count   <= '0;
          layer_done <= 1'b1;
        end else begin
          wr_count <= wr_count + 8'd1;
        end
      end
    end
  end

  // ---------------------------------------------------------------------------
  // Argmax over the logits (one logit index per cycle, all images in parallel)
  // ---------------------------------------------------------------------------
  logic [8*N-1:0] am_row;
  logic [8*N-1:0] best_val;
  logic [4*N-1:0] best_idx;

  assign am_row = logits[am_n];

  generate
    for (j = 0; j < N; j++) begin : g_argmax
      logic signed [7:0] cand, best;
      assign cand = am_row[8*j +: 8];
      assign best = best_val[8*j +: 8];
      always_ff @(posedge clk) begin
        if (state == S_ARGMAX && (am_n == 4'd0 || cand > best)) begin
          best_val[8*j +: 8] <= cand;
          best_idx[4*j +: 4] <= am_n;
        end
      end
    end
  endgenerate

  // ---------------------------------------------------------------------------
  // Results and performance counters
  // ---------------------------------------------------------------------------
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      done       <= 1'b0;
      cycles     <= '0;
      mac_cycles <= '0;
      runs       <= '0;
      preds      <= '0;
    end else begin
      if (state == S_IDLE && start) begin
        done       <= 1'b0;
        cycles     <= 32'd1;
        mac_cycles <= '0;
      end else if (busy) begin
        cycles <= cycles + 32'd1;
      end
      if (issue_valid)
        mac_cycles <= mac_cycles + 32'd1;
      if (state == S_DONE) begin           // busy falls on this same edge
        done  <= 1'b1;
        runs  <= runs + 32'd1;
        preds <= best_idx;
      end
    end
  end

  assign logit_row = logits[logit_sel[$clog2(P3)-1:0]];

endmodule
