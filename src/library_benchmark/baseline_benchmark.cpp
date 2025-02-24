#include "baseline_benchmark.h"
#include "utils.h"

int BaselineBenchmark::setup() noexcept {
  auto num_original_blocks_ = benchmark_config.computed.num_original_blocks;
  auto num_recovery_blocks_ = benchmark_config.computed.num_recovery_blocks;
  auto block_size_ = benchmark_config.block_size;

  baseline_init();
  
  uint8_t *original_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_original_blocks_));
  uint8_t *recovery_buffer_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, block_size_ * num_recovery_blocks_));

  if (!original_buffer_ || !recovery_buffer_) {
    std::cerr << "Leopard: Failed to allocate buffer(s).\n";
    teardown();
    return -1;
  }

  params_ = baseline_get_params(num_original_blocks_, num_recovery_blocks_, block_size_, original_buffer_, recovery_buffer_);

  InvMatPtr_ = static_cast<uint8_t*>(aligned_alloc(ALIGNMENT_BYTES, sizeof(uint32_t)*256*256));

  // Initialize data buffer with CRC blocks
  for (unsigned i = 0; i < num_original_blocks_; i++) {
    int write_res = write_validation_pattern(i, original_buffer_ + block_size_*i, block_size_);
    if (write_res) {
      std::cerr << "Leopard: Failed to write random checking packet.\n";
      teardown();
      return -1;
    }
  }
  
  return 0;
}

  



int BaselineBenchmark::encode() noexcept {
  std::cout << "Started Encode\n\n";
  baseline_encode(params_);
  std::cout << "Finished Encode\n\n";
  return 0;
}



int BaselineBenchmark::decode() noexcept {
  std::cout << "Started Decode\n\n";
  baseline_decode(params_, reinterpret_cast<uint32_t*>(InvMatPtr_), benchmark_config.num_lost_blocks, lost_block_idxs.data());
  std::cout << "Finished Decode\n\n";
  return 0;
}




bool BaselineBenchmark::check_for_corruption() const noexcept {

  if (benchmark_config.num_lost_blocks > 0) {
    std::cout << "Lost Blocks: [";

    for (unsigned i = 0; i < benchmark_config.num_lost_blocks; i++) {
      std::cout << " " << lost_block_idxs[i];
    }
    std::cout << " ]\n";
  }


  bool res = true;
  for (unsigned i = 0; i < benchmark_config.computed.num_original_blocks; i++) {
    if (!validate_block(static_cast<uint8_t*>(params_.orig_data) + i * benchmark_config.block_size, benchmark_config.block_size)) {
      uint32_t* block_ptr = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(params_.orig_data) + i * benchmark_config.block_size);

      std::cout << "Corruption in block " << i << '\n';
      std::cout << "Is:\n  ";
      for (unsigned i = 0; i < benchmark_config.block_size/32; i++) {
        std::cout << "0x" << std::hex << block_ptr[i] << " ";
      }

      std::cout << '\n';

      uint32_t *temp = (uint32_t*)malloc(benchmark_config.block_size);
      write_validation_pattern(i, (uint8_t*)temp, benchmark_config.block_size);

      std::cout << "Should be :\n  ";
      for (unsigned i = 0; i < benchmark_config.block_size/32; i++) {
        std::cout << "0x" << std::hex << temp[i] << " ";
      }
      std::cout << '\n';



      free(temp);
      


    

       

      res = false;
    } 
  }


  return res;
}



void BaselineBenchmark::simulate_data_loss() noexcept {
  for (unsigned i = 0; i < benchmark_config.num_lost_blocks; i++) {
    uint32_t idx = lost_block_idxs[i];

    if (idx < benchmark_config.computed.num_original_blocks) {
      // Zero out the block in the original data array, set the corresponding block pointer to nullptr
      memset(static_cast<uint8_t*>(params_.orig_data) + idx * benchmark_config.block_size, 0xFF, benchmark_config.block_size);
    } else {
      idx -= 127;
      // Zero out the block in the encoded data array, set the corresponding block pointer to nullptr
      memset(static_cast<uint8_t*>(params_.redundant_data) + idx * benchmark_config.block_size, 0, benchmark_config.block_size);
    }
  }
}



void BaselineBenchmark::teardown() noexcept {

}

void BaselineBenchmark::flush_cache() noexcept {
  
}