#ifndef XOREC_HPP
#define XOREC_HPP

#include "xorec_utils.hpp"
#include "utils.hpp"

#include <immintrin.h>
#define XOREC_AVX2 __m256i


void xorec_init();

/**
 * @brief Encodes data using XOR-based erasure coding.
 * @param data_buffer Pointer to the data buffer.
 * @param parity_buffer Pointer to the parity buffer.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @return XorecResult XorecResult indicating success or failure.
 */
XorecResult xorec_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
);


/**
 * @brief Decodes data using XOR-based erasure coding.
 * @param data_buffer Pointer to the data buffer.
 * @param parity_buffer Pointer to the parity buffer.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @param block_bitmap A bitmap indicating which blocks are present.
 * @return XorecResult XorecResult indicating success or failure.
 */
XorecResult xorec_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap
);



/**
 * @brief XORs two blocks of data with AVX2 SIMD instructions.
 * 
 * @param dest Pointer to the destination block.
 * @param src Pointer to the source block.
 * @param bytes Number of bytes to XOR.
 */
static void inline xorec_xor_blocks(void * XOREC_RESTRICT dest, const void * XOREC_RESTRICT src, size_t bytes) {
  XOREC_AVX2 * XOREC_RESTRICT dest256 = reinterpret_cast<XOREC_AVX2*>(__builtin_assume_aligned(dest, 64));
  const XOREC_AVX2 * XOREC_RESTRICT src256 = reinterpret_cast<const XOREC_AVX2*>(__builtin_assume_aligned(src, 64));
  for (; bytes >= 128; bytes -= 128, dest256 += 4, src256 += 4) {
    XOREC_AVX2 x0 = _mm256_xor_si256(_mm256_load_si256(dest256), _mm256_load_si256(src256));
    XOREC_AVX2 x1 = _mm256_xor_si256(_mm256_load_si256(dest256 + 1), _mm256_load_si256(src256 + 1));
    XOREC_AVX2 x2 = _mm256_xor_si256(_mm256_load_si256(dest256 + 2), _mm256_load_si256(src256 + 2));
    XOREC_AVX2 x3 = _mm256_xor_si256(_mm256_load_si256(dest256 + 3), _mm256_load_si256(src256 + 3));
    _mm256_store_si256(dest256, x0);
    _mm256_store_si256(dest256 + 1, x1);
    _mm256_store_si256(dest256 + 2, x2);
    _mm256_store_si256(dest256 + 3, x3);
  }
}
#endif // XOREC_HPP