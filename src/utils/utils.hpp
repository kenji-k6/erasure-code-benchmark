#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include "bm_reporters.hpp"


#define VALIDATION

/// @brief Constants for fixed values
constexpr size_t ALIGNMENT = 64;
constexpr size_t RANDOM_SEED = 1896;
constexpr size_t MIN_DATA_BLOCK_SIZE = 2;


/// @brief Constants for the XOR-EC algorithm
namespace ECLimits {
  constexpr size_t XOREC_BLOCK_ALIGNMENT = 64;

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
 * @class PCGRandom
 * @brief Implements the PCG random number generator for benchmarking.
 * 
 * @see https://www.pcg-random.org/
 */
class PCGRandom {
private:
  uint64_t state; ///< Internal RNG state
  uint64_t inc;   ///< Increment value (must be odd)

public:
  /**
   * @brief Construct a new PCGRandom object
   * 
   * @param seed 
   * @param seq 
   */
  PCGRandom(uint64_t seed, uint64_t seq);

  /**
   * @brief Returns the next random number in the sequence.
   * 
   * @return uint32_t 
   */
  uint32_t next();  ///< Generate a random 32-bit number
};


/**
 * @brief Writes a data pattern to a block for corruption detection.
 * @param block_idx Index of the block (used as a seed for reproducibility)
 * @param block_ptr Pointer to the block memory.
 * @param size Size of the block in bytes.
 * @return 0 on success, nonzero on failure.
 */
int write_validation_pattern(uint32_t block_idx, uint8_t* block_ptr, size_t bytes);


/**
 * @brief Check's if a block's content has been corrupted.
 * @param block_ptr Pointer to the block memory.
 * @param size Size of the block in bytes.
 * @return True if the block is not corrupted, false otherwise.
 */
bool validate_block(const uint8_t* block_ptr, size_t bytes);



void select_lost_block_idxs(size_t num_data_blocks, size_t num_parity_blocks, size_t num_lost_blocks, uint8_t* block_bitmap);

/**
 * @brief Helper function to throw an error message.
 * 
 * @param message 
 */
[[noreturn]] void throw_error(const std::string& message);


/**
 * @brief Helper function to convert a string to lowercase.
 * 
 * @param str 
 * @return std::string 
 */
std::string to_lower(std::string str);
#endif // UTILS_HPP