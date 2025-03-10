/**
 * @file isal_benchmark.cpp
 * @brief Benchmark implementation for the Intel ISA-L library's EC implementation
 * 
 * Documentation can be found in isal_benchmark.h and abstract_benchmark.h
 */


#include "isal_benchmark.h"

#include "erasure_code.h"
#include "utils.h"
#include <cstring>
#include <iostream>


ISALBenchmark::ISALBenchmark(const BenchmarkConfig& config) noexcept : ECBenchmark(config) {
  num_total_blocks_ = num_original_blocks_ + num_recovery_blocks_;
  // Allocate matrices etc.
  encode_matrix_ = std::make_unique<uint8_t[]>(num_total_blocks_ * num_original_blocks_);
  decode_matrix_ = std::make_unique<uint8_t[]>(num_total_blocks_ * num_original_blocks_);
  invert_matrix_ = std::make_unique<uint8_t[]>(num_total_blocks_ * num_original_blocks_);
  temp_matrix_ = std::make_unique<uint8_t[]>(num_total_blocks_ * num_original_blocks_);
  g_tbls_ = std::make_unique<uint8_t[]>(num_total_blocks_ * num_original_blocks_ * 32);

  original_buffer_ = std::make_unique<uint8_t[]>(block_size_ * num_total_blocks_);
  recovery_outp_buffer_ = std::make_unique<uint8_t[]>(block_size_ * num_recovery_blocks_);

  if (!encode_matrix_ || !decode_matrix_ || !invert_matrix_ || !temp_matrix_ || !g_tbls_ ||
      !original_buffer_ || !recovery_outp_buffer_) {
    throw_error("ISAL: Failed to allocate memory.");
  }

  // Initialize Pointer vectors
  original_ptrs_.resize(ECLimits::ISAL_MAX_TOT_BLOCKS);
  recovery_src_ptrs_.resize(ECLimits::ISAL_MAX_TOT_BLOCKS);
  recovery_outp_ptrs_.resize(ECLimits::ISAL_MAX_TOT_BLOCKS);
  decode_index_.resize(ECLimits::ISAL_MAX_TOT_BLOCKS);

  for (unsigned i = 0; i < num_total_blocks_; ++i) {
    original_ptrs_[i] = &original_buffer_[i * block_size_];
  }

  for (unsigned i = 0; i < num_original_blocks_; ++i) {
    recovery_outp_ptrs_[i] = &recovery_outp_buffer_[i * block_size_];
  }

  // Generate encode matricx, can be precomputed as it is fixed for a given num_original_blocks_
  gf_gen_cauchy1_matrix(encode_matrix_.get(), num_total_blocks_, num_original_blocks_);

  // Initialize generator tables for encoding, can be precomputed as it is fixed for a given num_original_blocks_
  ec_init_tables(num_original_blocks_, num_recovery_blocks_, &encode_matrix_[num_original_blocks_ * num_original_blocks_], g_tbls_.get());

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    int write_res = write_validation_pattern(i, original_ptrs_[i], block_size_);
    if (write_res) throw_error("ISAL: Failed to write random checking packet.");
  }
}

int ISALBenchmark::encode() noexcept {
  ec_encode_data(block_size_, num_original_blocks_, num_recovery_blocks_,
                 g_tbls_.get(), original_ptrs_.data(), original_ptrs_.data() + num_original_blocks_);
  return 0;
}



int ISALBenchmark::decode() noexcept {
  if (num_lost_blocks_ == 0) return 0;

  // Copy lost block indices (in reality this would be done by iterating and checking for lost blocks)
  for (auto idx : lost_block_idxs_) {
    block_err_list_.push_back(static_cast<uint8_t>(idx));
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
  ec_init_tables(num_original_blocks_, num_lost_blocks_, decode_matrix_.get(), g_tbls_.get());
  ec_encode_data(block_size_, num_original_blocks_, num_lost_blocks_,
                 g_tbls_.get(), recovery_src_ptrs_.data(), recovery_outp_ptrs_.data());
                 
  return 0;
}


void ISALBenchmark::simulate_data_loss() noexcept {
  for (auto idx : lost_block_idxs_) {
    memset(original_ptrs_[idx], 0, block_size_);
    original_ptrs_[idx] = nullptr;
    if (idx > num_original_blocks_) {
      memset(recovery_outp_ptrs_[idx - num_original_blocks_], 0, block_size_);
    }
  }
}


bool ISALBenchmark::check_for_corruption() const noexcept {
  unsigned loss_idx = 0;
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    bool res = false;
    if (loss_idx < num_lost_blocks_ && i == lost_block_idxs_[loss_idx]) {
      res = validate_block(recovery_outp_ptrs_[loss_idx], block_size_);
      loss_idx++;
    } else {
      res = validate_block(original_ptrs_[i], block_size_);
    }

    if (!res) return false;
  }
  return true;
}


int gf_gen_decode_matrix_simple(
  const std::unique_ptr<uint8_t[]>& encode_matrix,
  std::unique_ptr<uint8_t[]>& decode_matrix,
  std::unique_ptr<uint8_t[]>& invert_matrix,
  std::unique_ptr<uint8_t[]>& temp_matrix,
  std::vector<uint8_t>& decode_index,
  std::vector<uint8_t>& frag_err_list,
  const int nerrs, const int k, [[maybe_unused]] const int m
) {
  int i, j, p, r;
  int nsrcerrs = 0;
  uint8_t s;
  uint8_t frag_in_err[ECLimits::ISAL_MAX_TOT_BLOCKS];
  memset(frag_in_err, 0, sizeof(frag_in_err));

  // Order the fragments in erasure for easier sorting
  for (i = 0; i < nerrs; i++) {
    if (frag_err_list[i] < k) {
      nsrcerrs++;
    }
    frag_in_err[frag_err_list[i]] = 1;
  }
  // Construct b (matrix that encoded remaining blocks) by removing erased rows
  for (i = 0, r = 0; i < k; i++, r++) {
    while (frag_in_err[r]) {
      r++;
    }
    for (j = 0; j < k; j++) {
      
      temp_matrix[k * i + j] = encode_matrix[k * r + j];
    }

    decode_index[i] = r;
  }
  // Invert matrix to get recovery matrix
  if (gf_invert_matrix(temp_matrix.get(), invert_matrix.get(), k) < 0) {
    return -1;
  }
  // Get decode matrix with only wanted recovery rows
  for (i = 0; i < nerrs; i++) {
    if (frag_err_list[i] < k) {
      for (j = 0; j < k; j++) {
        decode_matrix[k * i + j] = invert_matrix[k * frag_err_list[i] + j];
      }
    }
  }
  // For non-src (parity) erasures need to multiply encode matrix * invert
  for (p = 0; p < nerrs; p++) {
    if (frag_err_list[p] >= k) { // A parity err
      for (i = 0; i < k; i++) {
        s = 0;
        for (j = 0; j < k; j++) {
          s ^= gf_mul(invert_matrix[j * k + i], encode_matrix[k * frag_err_list[p] + j]);
        }
        decode_matrix[k * p + i] = s;
      }
    }
  }
  return 0;
}