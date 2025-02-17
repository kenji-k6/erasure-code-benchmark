#include "isal_benchmark.h"


int ISALBenchmark::setup() {
  // Allocate matrices
  size_t tot_blocks = kConfig.computed.original_blocks + kConfig.computed.recovery_blocks;
  nerrs_ = 5; // TODO: adjust this accordingly

  encode_matrix_ = (uint8_t*) simd_safe_allocate(tot_blocks * kConfig.computed.original_blocks);
  decode_matrix_ = (uint8_t*) simd_safe_allocate(tot_blocks * kConfig.computed.original_blocks);
  invert_matrix_ = (uint8_t*) simd_safe_allocate(tot_blocks * kConfig.computed.original_blocks);
  temp_matrix_ = (uint8_t*) simd_safe_allocate(tot_blocks * kConfig.computed.original_blocks);
  g_tbls_ = (uint8_t*) simd_safe_allocate(tot_blocks * kConfig.computed.original_blocks * 32);

  if (!encode_matrix_ || !decode_matrix_ || !invert_matrix_ || !temp_matrix_ || !g_tbls_) {
    teardown();
    std::cerr << "ISAL: Failed to allocate matrices.\n";
    return -1;
  }

  // Allocate buffers
  original_data_ = (uint8_t*) simd_safe_allocate(kConfig.block_size * tot_blocks);
  recovery_outp_data_ = (uint8_t*) simd_safe_allocate(kConfig.block_size * kConfig.computed.recovery_blocks);

  if (!original_data_ || !recovery_outp_data_) {
    teardown();
    std::cerr << "ISAL: Failed to allocate buffers.\n";
    return -1;
  }

  // Initialize original data to 1s, recovery data to 0s
  memset(original_data_, 0xFF, kConfig.block_size * tot_blocks);
  memset(recovery_outp_data_, 0, kConfig.block_size * kConfig.computed.recovery_blocks);

  // Initialize pointers
  for (unsigned i = 0; i < tot_blocks; i++) {
    original_ptrs_[i] = (uint8_t*) original_data_ + (i * kConfig.block_size);
  }

  for (unsigned i = 0; i < kConfig.computed.recovery_blocks; i++) {
    recovery_outp_ptrs_[i] = (uint8_t*) recovery_outp_data_ + (i * kConfig.block_size);
  }


  // Generate encode matrix
  // A Cauchy matrix is a good choice as even large k are alrways inverable, keeping the recovery rule simple
  gf_gen_cauchy1_matrix(
    encode_matrix_,
    tot_blocks,
    kConfig.computed.original_blocks
  );

  // Initialize g_tbls_ from encode matrix
  ec_init_tables(
    kConfig.computed.original_blocks,
    kConfig.computed.recovery_blocks,
    &encode_matrix_[kConfig.computed.original_blocks * kConfig.computed.original_blocks],
    g_tbls_
  );

  // Generate a decode metrix
  int decode_matrix_result = gf_gen_decode_matrix_simple(
    encode_matrix_,
    decode_matrix_,
    invert_matrix_,
    temp_matrix_,
    decode_index,
    block_err_list_,
    nerrs_,
    kConfig.computed.original_blocks,
    tot_blocks
  );

  if (decode_matrix_result) {
    std::cerr << "ISAL: Failed to generate decode matrix.\n";
    return -1;
  }

  // Pack recovry array pointers as list of valid fragments
  for (unsigned i = 0; i < kConfig.computed.original_blocks; i++) {
    recovery_src_ptrs_[i] = original_ptrs_[decode_index[i]];
  }


  return 0;
}



void ISALBenchmark::teardown() {
  if (encode_matrix_) simd_safe_free(encode_matrix_);
  if (decode_matrix_) simd_safe_free(decode_matrix_);
  if (invert_matrix_) simd_safe_free(invert_matrix_);
  if (temp_matrix_) simd_safe_free(temp_matrix_);
  if (g_tbls_) simd_safe_free(g_tbls_);
  if (original_data_) simd_safe_free(original_data_);
  if (recovery_outp_data_) simd_safe_free(recovery_outp_data_);
}



int ISALBenchmark::encode() {
  ec_encode_data(
    kConfig.block_size,
    kConfig.computed.original_blocks,
    kConfig.computed.recovery_blocks,
    g_tbls_,
    original_ptrs_,
    &original_ptrs_[kConfig.computed.original_blocks]
  );

  return 0;
}



int ISALBenchmark::decode() {
  ec_init_tables(
    kConfig.computed.original_blocks,
    nerrs_,
    decode_matrix_,
    g_tbls_
  );

  ec_encode_data(
    kConfig.block_size,
    kConfig.computed.original_blocks,
    nerrs_,
    g_tbls_,
    recovery_src_ptrs_,
    recovery_outp_ptrs_
  );

  return 0;
}



void ISALBenchmark::flush_cache() {
  // TODO: Implement cache flushing
}



void ISALBenchmark::check_for_corruption() {
  // TODO: Implement corruption checking
}



void ISALBenchmark::simulate_data_loss() {
  // TODO: Implement data loss simulation
}




static int gf_gen_decode_matrix_simple(uint8_t *encode_matrix, uint8_t *decode_matrix, uint8_t *invert_matrix, uint8_t *temp_matrix, uint8_t *decode_index, uint8_t *frag_err_list, int nerrs, int k, int m) {
  int i, j, p, r;
  int nsrcerrs = 0;
  uint8_t s, *b = temp_matrix;
  uint8_t frag_in_err[ISAL_MAX_TOT_BLOCKS];
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