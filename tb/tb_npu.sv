// =============================================================================
// tb_npu.sv
// Full-system testbench for TinyNPU.
//
// For every batch of N MNIST test images it:
//   1. streams the 784 pixel beats into the AXI4-Stream port
//   2. writes CTRL = START | IRQ_EN over AXI4-Lite and waits for the interrupt
//   3. reads the predictions, all 10 logits of every image and the counters
//   4. compares every logit bit for bit with the Python golden model
//      (mem/golden.mem) and the prediction with the true label
//
// Plusargs
//   +NUM_IMAGES=<n>   images to run (default 10000)
//   +MEM_DIR=<path>   folder with test_images.mem and golden.mem (default ../../mem)
//   +GAPS             insert random idle cycles into the image stream
//   +VERBOSE          print every image
//   +SVA_SELFTEST     break the stream protocol once on purpose; passes only if
//                     the assertions catch it (needs NPU_SVA)
//
// Compile with +define+NPU_SVA together with tb/npu_sva.sv to include the
// assertion failure count in the result.
// =============================================================================
`timescale 1ns/1ps

module tb_npu;

  parameter int N = 8;

  localparam int IN_DIM        = 784;
  localparam int OUT_DIM       = 10;
  localparam int LINES_PER_IMG = IN_DIM / 16;
  localparam int MACS_PER_IMG  = 784 * 128 + 128 * 64 + 64 * 10;   // useful MACs
  localparam int TIMEOUT       = 2_000_000;                          // cycles per batch

  // register map
  localparam logic [11:0] REG_ID     = 12'h000;
  localparam logic [11:0] REG_CONFIG = 12'h004;
  localparam logic [11:0] REG_CTRL   = 12'h008;
  localparam logic [11:0] REG_STATUS = 12'h00C;
  localparam logic [11:0] REG_CYCLES = 12'h010;
  localparam logic [11:0] REG_MACCYC = 12'h014;
  localparam logic [11:0] REG_PRED0  = 12'h018;
  localparam logic [11:0] REG_PRED1  = 12'h01C;
  localparam logic [11:0] REG_RUNS   = 12'h020;
  localparam logic [11:0] REG_LOGITS = 12'h400;

  // ---------------------------------------------------------------------------
  // Clock, reset, DUT
  // ---------------------------------------------------------------------------
  logic aclk = 1'b0;
  logic aresetn = 1'b0;
  always #5 aclk = ~aclk;   // 100 MHz

  logic [11:0]    awaddr, araddr;
  logic           awvalid, awready, wvalid, wready, bvalid, bready;
  logic           arvalid, arready, rvalid, rready;
  logic [31:0]    wdata, rdata;
  logic [3:0]     wstrb;
  logic [1:0]     bresp, rresp;
  logic [8*N-1:0] tdata;
  logic           tvalid, tready, tlast;
  logic           irq;

  npu_top #(.N(N)) dut (
    .aclk           (aclk),
    .aresetn        (aresetn),
    .s_axil_awaddr  (awaddr),
    .s_axil_awvalid (awvalid),
    .s_axil_awready (awready),
    .s_axil_wdata   (wdata),
    .s_axil_wstrb   (wstrb),
    .s_axil_wvalid  (wvalid),
    .s_axil_wready  (wready),
    .s_axil_bresp   (bresp),
    .s_axil_bvalid  (bvalid),
    .s_axil_bready  (bready),
    .s_axil_araddr  (araddr),
    .s_axil_arvalid (arvalid),
    .s_axil_arready (arready),
    .s_axil_rdata   (rdata),
    .s_axil_rresp   (rresp),
    .s_axil_rvalid  (rvalid),
    .s_axil_rready  (rready),
    .s_axis_tdata   (tdata),
    .s_axis_tvalid  (tvalid),
    .s_axis_tready  (tready),
    .s_axis_tlast   (tlast),
    .irq            (irq)
  );

`ifdef NPU_SVA
  // ---------------------------------------------------------------------------
  // Assertions (tb/npu_sva.sv), attached to every npu_top / npu_core instance
  // ---------------------------------------------------------------------------
  bind npu_top npu_top_sva #(.N(N)) u_sva (
    .aclk   (aclk),             .aresetn(aresetn),
    .awaddr (s_axil_awaddr),    .awvalid(s_axil_awvalid), .awready(s_axil_awready),
    .wdata  (s_axil_wdata),     .wvalid (s_axil_wvalid),  .wready (s_axil_wready),
    .bresp  (s_axil_bresp),     .bvalid (s_axil_bvalid),  .bready (s_axil_bready),
    .araddr (s_axil_araddr),    .arvalid(s_axil_arvalid), .arready(s_axil_arready),
    .rdata  (s_axil_rdata),     .rresp  (s_axil_rresp),
    .rvalid (s_axil_rvalid),    .rready (s_axil_rready),
    .tdata  (s_axis_tdata),     .tvalid (s_axis_tvalid),  .tready (s_axis_tready),
    .tlast  (s_axis_tlast),
    .start  (start),            .busy   (busy),           .done   (done),
    .irq_en (irq_en),           .irq    (irq)
  );

  bind npu_core npu_core_sva #(.N(N)) u_sva (
    .clk        (clk),
    .rst_n      (rst_n),
    .state      (state),
    .issue_valid(issue_valid),
    .k          (k),
    .k_len      (k_len),
    .g_valid    (g_valid),
    .rq_valid0  (rq_valid[0]),
    .w5_neuron  (w5_neuron),
    .p_len      (p_len),
    .layer_done (layer_done)
  );
`endif

  // ---------------------------------------------------------------------------
  // Test data
  // ---------------------------------------------------------------------------
  bit [7:0]  pixels  [0:N-1][0:IN_DIM-1];
  int        labels  [0:N-1];
  int        g_preds [0:N-1];
  bit [79:0] g_logits[0:N-1];

  int    fd_img, fd_gold;
  int    errors = 0;
  bit    gaps, verbose, sva_selftest;

  // ---------------------------------------------------------------------------
  // AXI4-Lite master (BREADY and RREADY are always high)
  // ---------------------------------------------------------------------------
  task automatic axil_write(input logic [11:0] addr, input logic [31:0] data,
                            output logic [1:0] resp);
    bit aw_done = 0, w_done = 0;
    @(posedge aclk);
    awaddr  <= addr;
    awvalid <= 1'b1;
    wdata   <= data;
    wstrb   <= 4'hF;
    wvalid  <= 1'b1;
    while (!(aw_done && w_done)) begin
      @(posedge aclk);
      if (awvalid && awready) begin aw_done = 1; awvalid <= 1'b0; end
      if (wvalid  && wready)  begin w_done  = 1; wvalid  <= 1'b0; end
    end
    while (!bvalid) @(posedge aclk);
    resp = bresp;
  endtask

  task automatic axil_read(input logic [11:0] addr, output logic [31:0] data,
                           output logic [1:0] resp);
    @(posedge aclk);
    araddr  <= addr;
    arvalid <= 1'b1;
    do @(posedge aclk); while (!arready);
    arvalid <= 1'b0;
    while (!rvalid) @(posedge aclk);
    data = rdata;      // R handshake completes on this edge (RREADY is high)
    resp = rresp;
  endtask

  task automatic expect_reg(input logic [11:0] addr, input logic [31:0] exp, input string name);
    logic [31:0] d;
    logic [1:0]  r;
    axil_read(addr, d, r);
    if (d !== exp || r !== 2'b00) begin
      $display("ERROR: %s = 0x%08h (resp %0d), expected 0x%08h", name, d, r, exp);
      errors++;
    end
  endtask

  task automatic read_batch(input int count);
    bit [127:0] line;
    bit [3:0]   lbl, prd;
    bit [79:0]  lg;
    int         rc;
    for (int j = 0; j < N; j++) begin
      for (int q = 0; q < LINES_PER_IMG; q++) begin
        if (j < count) begin
          rc = $fscanf(fd_img, "%h\n", line);
          if (rc != 1) $fatal(1, "could not read test_images.mem");
        end else begin
          line = '0;       // unused lanes of the last batch
        end
        for (int b = 0; b < 16; b++) pixels[j][16*q + b] = line[8*b +: 8];
      end
      if (j < count) begin
        rc = $fscanf(fd_gold, "%h %h %h\n", lbl, prd, lg);
        if (rc != 3) $fatal(1, "could not read golden.mem");
        labels[j]   = lbl;
        g_preds[j]  = prd;
        g_logits[j] = lg;
      end
    end
  endtask

  task automatic stream_batch();
    @(posedge aclk);
    for (int k = 0; k < IN_DIM; k++) begin
      if (gaps && ($urandom % 4 == 0)) begin
        tvalid <= 1'b0;
        repeat ($urandom % 3 + 1) @(posedge aclk);
      end
      for (int j = 0; j < N; j++) tdata[8*j +: 8] <= pixels[j][k];
      tvalid <= 1'b1;
      tlast  <= (k == IN_DIM - 1);
      do @(posedge aclk); while (!tready);
    end
    tvalid <= 1'b0;
    tlast  <= 1'b0;
  endtask

  // ---------------------------------------------------------------------------
  // Main
  // ---------------------------------------------------------------------------
  initial begin
    int          num_images, done_images, batch, count, correct;
    int          pred_mismatch, logit_mismatch, wait_cycles;
    longint      total_cycles, total_mac_cycles;
    logic [31:0] d, cyc, maccyc, pred0, pred1;
    logic [1:0]  r;
    string       mem_dir;
    real         cyc_per_img, util;
    int          sva_failures;

    awaddr = '0; awvalid = 0; wdata = '0; wstrb = '0; wvalid = 0; bready = 1;
    araddr = '0; arvalid = 0; rready = 1;
    tdata = '0; tvalid = 0; tlast = 0;

    if (!$value$plusargs("NUM_IMAGES=%d", num_images)) num_images = 10000;
    if (!$value$plusargs("MEM_DIR=%s", mem_dir))       mem_dir = "../../mem";
    gaps         = $test$plusargs("GAPS");
    verbose      = $test$plusargs("VERBOSE");
    sva_selftest = $test$plusargs("SVA_SELFTEST");

    fd_img  = $fopen({mem_dir, "/test_images.mem"}, "r");
    fd_gold = $fopen({mem_dir, "/golden.mem"}, "r");
    if (fd_img == 0 || fd_gold == 0) $fatal(1, "cannot open test vectors in %s", mem_dir);

    $display("");
    $display("==== TinyNPU system test: N = %0d (%0dx%0d array), %0d images ====", N, N, N, num_images);

    repeat (5) @(posedge aclk);
    aresetn <= 1'b1;
    repeat (2) @(posedge aclk);

    // register sanity checks
    expect_reg(REG_ID,     32'h4E50_5531, "ID");
    expect_reg(REG_CONFIG, {16'd784, 8'd3, 8'(N)}, "CONFIG");
    expect_reg(REG_STATUS, 32'h0, "STATUS after reset");
    axil_write(12'h010, 32'h1234, r);
    if (r !== 2'b10) begin $display("ERROR: write to read-only CYCLES returned %0d", r); errors++; end
    axil_read(12'h3F0, d, r);
    if (r !== 2'b10) begin $display("ERROR: unmapped read returned %0d", r); errors++; end

    done_images = 0;
    correct = 0;
    pred_mismatch = 0;
    logit_mismatch = 0;
    total_cycles = 0;
    total_mac_cycles = 0;
    batch = 0;

    while (done_images < num_images) begin
      count = (num_images - done_images < N) ? (num_images - done_images) : N;
      read_batch(count);
      stream_batch();

      axil_read(REG_STATUS, d, r);
      if (d[2] !== 1'b1 || d[3] !== 1'b0) begin
        $display("ERROR: batch %0d: STATUS = 0x%08h after streaming (IMG_READY should be 1)", batch, d);
        errors++;
      end

      axil_write(REG_CTRL, 32'h3, r);     // START + IRQ_EN
      // DONE (and so irq) of the previous batch clears when START is taken
      wait_cycles = 0;
      do begin
        @(posedge aclk);
        wait_cycles++;
        if (wait_cycles > 10) $fatal(1, "batch %0d: START was not accepted", batch);
      end while (irq);

      // deliberate protocol violation: change TDATA while TREADY is low
      if (sva_selftest && batch == 0) begin
        tvalid <= 1'b1;
        tdata  <= '1;
        @(posedge aclk);
        tdata  <= '0;
        @(posedge aclk);
        tvalid <= 1'b0;
      end
      wait_cycles = 0;
      while (!irq) begin
        @(posedge aclk);
        wait_cycles++;
        if (wait_cycles > TIMEOUT) $fatal(1, "batch %0d: no interrupt after %0d cycles", batch, TIMEOUT);
      end

      axil_read(REG_CYCLES, cyc, r);
      axil_read(REG_MACCYC, maccyc, r);
      axil_read(REG_PRED0, pred0, r);
      axil_read(REG_PRED1, pred1, r);
      total_cycles     += cyc;
      total_mac_cycles += maccyc;

      for (int j = 0; j < count; j++) begin
        int        pred;
        bit [79:0] got;
        pred = (j < 8) ? int'(pred0[4*j +: 4]) : int'(pred1[4*(j-8) +: 4]);
        for (int n = 0; n < OUT_DIM; n++) begin
          axil_read(REG_LOGITS + 12'(64 * j + 4 * n), d, r);
          got[8*n +: 8] = d[7:0];
          if (d[31:8] !== {24{d[7]}}) begin
            $display("ERROR: logit read not sign-extended: 0x%08h", d);
            errors++;
          end
        end
        if (got !== g_logits[j]) begin
          logit_mismatch++;
          if (logit_mismatch <= 10)
            $display("MISMATCH image %0d: logits %h, golden %h", done_images + j, got, g_logits[j]);
        end
        if (pred != g_preds[j]) begin
          pred_mismatch++;
          if (pred_mismatch <= 10)
            $display("MISMATCH image %0d: prediction %0d, golden %0d", done_images + j, pred, g_preds[j]);
        end
        if (pred == labels[j]) correct++;
        if (verbose)
          $display("image %5d  label %0d  predicted %0d  %s", done_images + j, labels[j], pred,
                   (pred == labels[j]) ? "" : "(wrong)");
      end

      done_images += count;
      batch++;
      if (done_images % 1000 == 0 || done_images == num_images)
        $display("  %5d images done, accuracy so far %0.2f%%, mismatches %0d",
                 done_images, 100.0 * correct / done_images, logit_mismatch + pred_mismatch);
    end

    expect_reg(REG_RUNS, 32'(batch), "RUNS");

`ifdef NPU_SVA
    sva_failures = dut.u_sva.failures + dut.u_core.u_sva.failures;
`else
    sva_failures = -1;
`endif

    cyc_per_img = real'(total_cycles) / num_images;
    util = 100.0 * real'(MACS_PER_IMG) * num_images / (real'(total_cycles) * N * N);

    $display("");
    $display("==== Summary ====");
    $display("  Array                  : %0d x %0d (%0d MAC units)", N, N, N * N);
    $display("  Images                 : %0d", num_images);
    $display("  Accuracy (vs labels)   : %0.2f%% (%0d / %0d)", 100.0 * correct / num_images, correct, num_images);
    $display("  Logit mismatches       : %0d (vs bit-exact golden model)", logit_mismatch);
    $display("  Prediction mismatches  : %0d", pred_mismatch);
    $display("  Cycles per batch       : %0.1f", real'(total_cycles) / batch);
    $display("  Cycles per image       : %0.1f", cyc_per_img);
    $display("  Array utilization      : %0.1f%% (useful MACs / (cycles x %0d))", util, N * N);
    $display("  Feed efficiency        : %0.1f%% (cycles the array received data)",
             100.0 * real'(total_mac_cycles) / real'(total_cycles));
    $display("  Throughput @ 100 MHz   : %0.0f images/s", 1.0e8 / cyc_per_img);
    if (sva_failures >= 0)
      $display("  Assertion failures     : %0d", sva_failures);
    else
      $display("  Assertion failures     : (assertions not compiled)");
    $display("RESULT N=%0d images=%0d correct=%0d logit_mismatch=%0d pred_mismatch=%0d cycles=%0d mac_cycles=%0d batches=%0d",
             N, num_images, correct, logit_mismatch, pred_mismatch, total_cycles, total_mac_cycles, batch);
    if (sva_selftest) begin
      if (sva_failures > 0)
        $display("*** SVA SELF-TEST PASSED *** (the assertions caught the injected violation)");
      else
        $display("*** SVA SELF-TEST FAILED *** (the injected violation was not caught)");
    end else if (errors == 0 && logit_mismatch == 0 && pred_mismatch == 0 && sva_failures <= 0)
      $display("*** TEST PASSED ***");
    else
      $display("*** TEST FAILED *** (%0d errors)",
               errors + logit_mismatch + pred_mismatch + (sva_failures > 0 ? sva_failures : 0));
    $fclose(fd_img);
    $fclose(fd_gold);
    $finish;
  end

endmodule
