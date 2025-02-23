#include "baseline_benchmark.h"
#include "utils.h"

#define NUM_LOST_TEMP 0



int BaselineBenchmark::setup() noexcept {
  
  encoder_ = rs_create_encoder(127, 127, 4096);
  decoder_ = rs_create_decoder(127, 127, 4096, 127-NUM_LOST_TEMP);
  inv_mat = new uint32_t[127 * 4096];
  orig_data = new uint8_t[127 * 4096];
  redundant_data = new uint8_t[127 * 4096];

  rs_init_tables();

  // Write Data
  for (int i = 0; i < 127; i++) {
    write_validation_pattern(i, orig_data + i * 4096, 4096);
  }

  int temp_lost_idx = 0;
  for (int i = 0; i < 256; i++) {
    if (benchmark_config.num_lost_blocks > temp_lost_idx && lost_block_idxs[temp_lost_idx] == i) {
      recv_idx[i] = 0;
      temp_lost_idx++;
    } else {
      recv_idx[i] = 1;
    }
  }
  return 0;
}



void BaselineBenchmark::teardown() noexcept {

}
  



int BaselineBenchmark::encode() noexcept {
  rs_encode(encoder_, orig_data, redundant_data);
  return 0;
}



int BaselineBenchmark::decode() noexcept {


  rs_decode(decoder_, orig_data, redundant_data, orig_data, recv_idx, inv_mat);
  return 0;
}



void BaselineBenchmark::flush_cache() noexcept {
  
}



bool BaselineBenchmark::check_for_corruption() const noexcept {
  for (int i = 0; i < 127; i++) {
    if (!validate_block(orig_data + i * 4096, 4096)) {
      return false;
    }
  }
  return true; 
}



void BaselineBenchmark::simulate_data_loss() noexcept {
  for (unsigned i = 0; i < NUM_LOST_TEMP; i++) {
    uint32_t idx = lost_block_idxs[i];
    if (idx < 127) {
      // Zero out the block in the original data array, set the corresponding block pointer to nullptr
      memset(orig_data + idx * 4096, 0, 4096);
    } else {
      idx -= 127;
      // Zero out the block in the encoded data array, set the corresponding block pointer to nullptr
      memset(redundant_data + idx * 4096, 0, 4096);
    }
  }
}