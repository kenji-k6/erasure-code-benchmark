#include "isal_benchmark.h"

#include "erasure_code.h"
#include <cstring>
#include <iostream>



/**
 * @file isal_benchmark.cpp
 * @brief Benchmark implementation for the Intel ISA-L library's ECC implementation
 * 
 * Documentation can be found in isal_benchmark.h and abstract_benchmark.h
 */


int ISALBenchmark::setup() noexcept {
  // Store frequently used variables / variables used in performance-critical areas locally
  num_original_blocks_ = benchmark_config.computed.num_original_blocks;
  num_recovery_blocks_ = benchmark_config.computed.num_recovery_blocks;
  num_total_blocks_ = num_original_blocks_ + num_recovery_blocks_;
  block_size_ = benchmark_config.block_size;
  num_lost_blocks_ = benchmark_config.num_lost_blocks;

  // Allocate matrices with aligned memory
  encode_matrix_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, num_total_blocks_ * num_original_blocks_));
  decode_matrix_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, num_total_blocks_ * num_original_blocks_));
  invert_matrix_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, num_total_blocks_ * num_original_blocks_));
  temp_matrix_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, num_total_blocks_ * num_original_blocks_));
  g_tbls_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, num_total_blocks_ * num_original_blocks_ * 32));

  if (!encode_matrix_ || !decode_matrix_ || !invert_matrix_ || !temp_matrix_ || !g_tbls_) {
    std::cerr << "ISAL: Failed to allocate matrices.\n";
    teardown();
    return -1;
  }

  // Allocate aligned buffers for data
  original_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_total_blocks_));
  recovery_outp_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_recovery_blocks_));

  if (!original_buffer_ || !recovery_outp_buffer_) {
    std::cerr << "ISAL: Failed to allocate data buffers.\n";
    teardown();
    return -1;
  }

  // Allocate pointers to the data blocks
  for (unsigned i = 0; i < num_total_blocks_; i++) {
    original_ptrs_[i] = original_buffer_ + i * block_size_;
  }

  for (unsigned i = 0; i < num_recovery_blocks_; i++) {
    recovery_outp_ptrs_[i] = recovery_outp_buffer_ + i * block_size_;
  }

  // Generate encode matrix, can be precomputed as it is fixed for a given num_original_blocks_
  gf_gen_cauchy1_matrix(encode_matrix_, num_total_blocks_, num_original_blocks_);

  // Initialize generator tables for encoding, can be precomputed as it is fixed for a given num_original_blocks_
  ec_init_tables(num_original_blocks_, num_recovery_blocks_,
                 &encode_matrix_[num_original_blocks_ * num_original_blocks_], g_tbls_);
  
  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    int write_res = write_validation_pattern(i, original_ptrs_[i], block_size_);
    if (write_res) {
      std::cerr << "ISAL: Failed to write random checking packet.\n";
      teardown();
      return -1;
    }
  }
  
  return 0;
}


void ISALBenchmark::teardown() noexcept {
  if (encode_matrix_) free(encode_matrix_);
  if (decode_matrix_) free(decode_matrix_);
  if (invert_matrix_) free(invert_matrix_);
  if (temp_matrix_) free(temp_matrix_);
  if (g_tbls_) free(g_tbls_);
  if (original_buffer_) free(original_buffer_);
  if (recovery_outp_buffer_) free(recovery_outp_buffer_);
}


int ISALBenchmark::encode() noexcept {
  ec_encode_data(block_size_, num_original_blocks_, num_recovery_blocks_, g_tbls_,
                 original_ptrs_, &original_ptrs_[num_original_blocks_]);
  return 0;
}


int ISALBenchmark::decode() noexcept {
  // Copy lost block indices (in reality this would be done by iterating and checking for lost blocks)
  for (unsigned i = 0; i < num_lost_blocks_; i++) {
    block_err_list_[i] = static_cast<uint8_t>(lost_block_idxs[i]);
  }

  // Generate decoding matrix
  if (gf_gen_decode_matrix_simple(encode_matrix_, decode_matrix_, invert_matrix_,
                                  temp_matrix_, decode_index_, block_err_list_,
                                  num_lost_blocks_, num_original_blocks_, num_total_blocks_)) {
    return -1;
  }

  // Set up recovery pointers
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    recovery_src_ptrs_[i] = original_ptrs_[decode_index_[i]];
  }

  // Initialize tables and perform recovery
  ec_init_tables(num_original_blocks_, num_lost_blocks_, decode_matrix_, g_tbls_);
  ec_encode_data(block_size_, num_original_blocks_, num_lost_blocks_, g_tbls_,
                 recovery_src_ptrs_, recovery_outp_ptrs_);
  
  return 0;
}


void ISALBenchmark::simulate_data_loss() noexcept {
  for (unsigned i = 0; i < num_lost_blocks_; i++) {
    uint32_t idx = lost_block_idxs[i];
    // Zero out the lost block, set the corresponding block pointer to nullptr
    memset(original_ptrs_[idx], 0, block_size_);
    original_ptrs_[idx] = nullptr;

    if (idx > num_original_blocks_) { // If the lost block is a recovery block, zero out the recovery output as well
      memset(recovery_outp_ptrs_[idx - num_original_blocks_], 0, block_size_);
    }
  }
}


bool ISALBenchmark::check_for_corruption() const noexcept {
  unsigned loss_idx = 0;
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    bool res = false;
    if (i == lost_block_idxs[loss_idx]) {
      res = validate_block(recovery_outp_ptrs_[loss_idx], block_size_);
      loss_idx++;
    } else {
      res = validate_block(original_ptrs_[i], block_size_);
    }

    if (!res) return false;
  }
  return true;
}


void ISALBenchmark::flush_cache() noexcept {
  // TODO: Implement cache flushing
}


static int gf_gen_decode_matrix_simple(uint8_t *encode_matrix, uint8_t *decode_matrix, uint8_t *invert_matrix,
                                       uint8_t *temp_matrix, uint8_t *decode_index, uint8_t *frag_err_list,
                                       int nerrs, int k, int m) {
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
  
  // Construct b (matrix that encoded remaining blocks) by removing erased rows
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