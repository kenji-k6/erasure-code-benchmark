#ifndef UTILS_H
#define UTILS_H

#include <cstddef>
#include <cstdint>
#include <vector>


// Constants
constexpr size_t ALIGNMENT_BYTES = 64;
constexpr size_t RANDOM_SEED = 1896;
constexpr size_t MIN_DATA_BLOCK_SIZE = 2;


// Error Correction Code (ECC) constraints
namespace ECCLimits {
  constexpr size_t BASELINE_BLOCK_ALIGNMENT = 64;

  constexpr size_t CM256_MAX_TOT_BLOCKS = 256;

  constexpr size_t ISAL_MIN_BLOCK_SIZE = 64;
  constexpr size_t ISAL_MAX_DATA_BLOCKS = 256;
  constexpr size_t ISAL_MAX_TOT_BLOCKS = 256;

  constexpr size_t LEOPARD_MAX_TOT_BLOCKS = 65'536;
  constexpr size_t LEOPARD_BLOCK_ALIGNMENT = 64;

  constexpr size_t WIREHAIR_MIN_DATA_BLOCKS = 2;
  constexpr size_t WIREHAIR_MAX_DATA_BLOCKS = 64'000;
}


/**
 * @struct BenchmarkConfig
 * @brief Configuration parameters for the benchmark
 */
struct BenchmarkConfig {
  uint64_t data_size;               ///< Total size of original data
  uint64_t block_size;              ///< Size of each block

  uint64_t num_lost_blocks;         ///< Number of total blocks lost (recovery + original)
  uint32_t *lost_block_idxs;        ///< Pointer to the lost block indices array
  double redundancy_ratio;          ///< Recovery blocks / original blocks ratio

  struct ComputedValues {
    uint32_t num_original_blocks;   ///< Number of original data blocks
    uint32_t num_recovery_blocks;   ///< Number of recovery blocks
  } computed;

  uint32_t num_iterations;
  uint8_t plot_id;          ///< Identifier for plotting 0
};


/**
 * @class PCGRandom
 * @brief Implements the PCG random number generator for benchmarking.
 */
class PCGRandom {
private:
  uint64_t state; ///< Internal RNG state
  uint64_t inc;   ///< Increment value (must be odd)

public:
  PCGRandom(uint64_t seed, uint64_t seq);
  uint32_t next();  ///< Generate a random 32-bit number
};


/**
 * @brief Writes a data pattern to a block for corruption detection.
 * @param block_idx Index of the block (used as a seed for reproducibility)
 * @param block_ptr Pointer to the block memory.
 * @param size Size of the block in bytes.
 * @return 0 on success, nonzero on failure.
 */
int write_validation_pattern(size_t block_idx, uint8_t* block_ptr, uint32_t size);


/**
 * @brief Check's if a block's content has been corrupted.
 * @param block_ptr Pointer to the block memory.
 * @param size Size of the block in bytes.
 * @return True if the block is not corrupted, false otherwise.
 */
bool validate_block(const uint8_t* block_ptr, uint32_t size);


/**
 * @brief Selects k unique indices from range [0, maxIndex) to determine lost blocks.
 * @param num_lost_blocks Number of blocks to select.
 * @param max_index Upper limit of the index range.
 * @param lost_block_idxs Vector to store the selected lost block indices.
 */
void select_lost_block_idxs(size_t num_lost_blocks, size_t max_idx, uint32_t *lost_block_idxs);

#endif // UTILS_H