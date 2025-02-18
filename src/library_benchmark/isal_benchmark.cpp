#include "isal_benchmark.h"


int ISALBenchmark::setup() noexcept {
  // Allocate matrices
  size_t tot_blocks = benchmark_config.computed.num_original_blocks + benchmark_config.computed.num_recovery_blocks;

  encode_matrix_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, tot_blocks * benchmark_config.computed.num_original_blocks);
  decode_matrix_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, tot_blocks * benchmark_config.computed.num_original_blocks);
  invert_matrix_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, tot_blocks * benchmark_config.computed.num_original_blocks);
  temp_matrix_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, tot_blocks * benchmark_config.computed.num_original_blocks);
  g_tbls_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, tot_blocks * benchmark_config.computed.num_original_blocks * 32);

  if (!encode_matrix_ || !decode_matrix_ || !invert_matrix_ || !temp_matrix_ || !g_tbls_) {
    teardown();
    std::cerr << "ISAL: Failed to allocate matrices.\n";
    return -1;
  }

  // copy lost block indices to frag err list
  for (unsigned i = 0; i < benchmark_config.num_lost_blocks; i++) {
    block_err_list_[i] = (uint8_t) lost_block_idxs[i];
  }

  // Allocate buffers
  original_data_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, benchmark_config.block_size * tot_blocks);
  recovery_outp_data_ = (uint8_t*) aligned_alloc(ALIGNMENT_BYTES, benchmark_config.block_size * benchmark_config.computed.num_recovery_blocks);

  if (!original_data_ || !recovery_outp_data_) {
    teardown();
    std::cerr << "ISAL: Failed to allocate buffers.\n";
    return -1;
  }

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    int write_res = write_validation_pattern(
      i,
      (uint8_t *) original_data_ + (i * benchmark_config.block_size),
      benchmark_config.block_size
    );

    if (write_res) {
      teardown();
      std::cerr << "ISAL: Failed to write random checking packet.\n";
      return -1;
    }
  }

  // Initialize pointers
  for (unsigned i = 0; i < tot_blocks; i++) {
    original_ptrs_[i] = (uint8_t*) original_data_ + (i * benchmark_config.block_size);
  }

  for (unsigned i = 0; i < benchmark_config.computed.num_recovery_blocks; i++) {
    recovery_outp_ptrs_[i] = (uint8_t*) recovery_outp_data_ + (i * benchmark_config.block_size);
  }


  // Generate encode matrix
  // A Cauchy matrix is a good choice as even large k are alrways inverable, keeping the recovery rule simple
  gf_gen_cauchy1_matrix(
    encode_matrix_,
    tot_blocks,
    benchmark_config.computed.num_original_blocks
  );

  // Initialize g_tbls_ from encode matrix
  ec_init_tables(
    benchmark_config.computed.num_original_blocks,
    benchmark_config.computed.num_recovery_blocks,
    &encode_matrix_[benchmark_config.computed.num_original_blocks * benchmark_config.computed.num_original_blocks],
    g_tbls_
  );

  return 0;
}



void ISALBenchmark::teardown() noexcept {
  if (encode_matrix_) free(encode_matrix_);
  if (decode_matrix_) free(decode_matrix_);
  if (invert_matrix_) free(invert_matrix_);
  if (temp_matrix_) free(temp_matrix_);
  if (g_tbls_) free(g_tbls_);
  if (original_data_) free(original_data_);
  if (recovery_outp_data_) free(recovery_outp_data_);
}



int ISALBenchmark::encode() noexcept {
  ec_encode_data(
    benchmark_config.block_size,
    benchmark_config.computed.num_original_blocks,
    benchmark_config.computed.num_recovery_blocks,
    g_tbls_,
    original_ptrs_,
    &original_ptrs_[benchmark_config.computed.num_original_blocks]
  );

  return 0;
}



int ISALBenchmark::decode() noexcept {
  size_t tot_blocks = benchmark_config.computed.num_original_blocks + benchmark_config.computed.num_recovery_blocks;
  // Generate a decode metrix
  int decode_matrix_result = gf_gen_decode_matrix_simple(
    encode_matrix_,
    decode_matrix_,
    invert_matrix_,
    temp_matrix_,
    decode_index,
    block_err_list_,
    benchmark_config.num_lost_blocks,
    benchmark_config.computed.num_original_blocks,
    tot_blocks
  );

  if (decode_matrix_result) {
    return -1;
  }

  // Pack recovry array pointers as list of valid fragments
  for (unsigned i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    recovery_src_ptrs_[i] = original_ptrs_[decode_index[i]];
  }

  ec_init_tables(
    benchmark_config.computed.num_original_blocks,
    benchmark_config.num_lost_blocks,
    decode_matrix_,
    g_tbls_
  );

  ec_encode_data(
    benchmark_config.block_size,
    benchmark_config.computed.num_original_blocks,
    benchmark_config.num_lost_blocks,
    g_tbls_,
    recovery_src_ptrs_,
    recovery_outp_ptrs_
  );

  return 0;
}



void ISALBenchmark::flush_cache() noexcept {
  // TODO: Implement cache flushing
}



bool ISALBenchmark::check_for_corruption() const noexcept {
  unsigned loss_idx = 0;

  for (unsigned i = 0; i < benchmark_config.computed.num_original_blocks; i++) {

    if (i == lost_block_idxs[loss_idx]) {
       if (!(validate_block(recovery_outp_ptrs_[loss_idx], benchmark_config.block_size))) return false;
       loss_idx++;
    } else {
      if (!validate_block(original_ptrs_[i], benchmark_config.block_size)) return false;
    }
  }
  return true;
}



void ISALBenchmark::simulate_data_loss() noexcept {
  // TODO: Implement data loss simulation
  for (unsigned i = 0; i < benchmark_config.num_lost_blocks; i++) {
    uint32_t idx = lost_block_idxs[i];
    // Zero out the lost block
    memset(original_ptrs_[idx], 0, benchmark_config.block_size);
    original_ptrs_[idx] = nullptr;

    if (idx > benchmark_config.computed.num_original_blocks) {
      memset(recovery_outp_ptrs_[idx - benchmark_config.computed.num_original_blocks], 0, benchmark_config.block_size);
    }
  }
}




static int gf_gen_decode_matrix_simple(uint8_t *encode_matrix, uint8_t *decode_matrix, uint8_t *invert_matrix, uint8_t *temp_matrix, uint8_t *decode_index, uint8_t *frag_err_list, int nerrs, int k, int m) {
  int i, j, p, r;
  int nsrcerrs = 0;
  uint8_t s, *b = temp_matrix;
  uint8_t frag_in_err[ECCLimits::ISAL_MAX_TOT_BLOCKS];
  memset(frag_in_err, 0, sizeof(frag_in_err));
  
  // Order the fragments in erasure for easier sorting
  for (i = 0; i < nerrs; i++) {
          if (frag_err_list[i] < k)
                  nsrcerrs++;
          frag_in_err[frag_err_list[i]] = 1;
  }
  
  // Construct b (matrix that encoded remaining frags) by removing erased rows
  for (i = 0, r = 0; i < k; i++, r++) {
          while (frag_in_err[r])
                  r++;
          for (j = 0; j < k; j++)
                  b[k * i + j] = encode_matrix[k * r + j];
          decode_index[i] = r;
  }
  
  // Invert matrix to get recovery matrix
  if (gf_invert_matrix(b, invert_matrix, k) < 0)
          return -1;

  // Get decode matrix with only wanted recovery rows
  for (i = 0; i < nerrs; i++) {
          if (frag_err_list[i] < k) // A src err
                  for (j = 0; j < k; j++)
                          decode_matrix[k * i + j] = invert_matrix[k * frag_err_list[i] + j];
  }
  
  // For non-src (parity) erasures need to multiply encode matrix * invert
  for (p = 0; p < nerrs; p++) {
          if (frag_err_list[p] >= k) { // A parity err
                  for (i = 0; i < k; i++) {
                          s = 0;
                          for (j = 0; j < k; j++)
                                  s ^= gf_mul(invert_matrix[j * k + i],
                                              encode_matrix[k * frag_err_list[p] + j]);
                          decode_matrix[k * p + i] = s;
                  }
          }
  }
  return 0;
}