#ifndef XOREC_HPP
#define XOREC_HPP

#include "xorec_utils.hpp"
#include "utils.hpp"


/**
 * @file xorec.h
 * @brief Provides encoding and decoding functions for custom XOR-based erasure coding.
 * 
 * This header defines the XOR-based erasure encoding and decoding functions,
 * optimized with SIMD intrinsics when available. It supports AVX, AVX2 and AVX512
 */



#if defined(__AVX__)
  #define TRY_XOREC_AVX
  #include <immintrin.h>
  #define XOREC_AVX __m128i
#endif

#if defined(__AVX2__)
  #define TRY_XOREC_AVX2
  #include <immintrin.h>
  #define XOREC_AVX2 __m256i
#endif

#if defined(__AVX512F__)
  #define TRY_XOREC_AVX512
  #include <immintrin.h>
  #define XOREC_AVX512 __m512i
#endif

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
  size_t num_parity_blocks,
  XorecVersion version
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
  const uint8_t * XOREC_RESTRICT block_bitmap, ///< Indexing for parity blocks starts at bit 128, e.g. the j-th parity block is at bit 128 + j, j < 128
  XorecVersion version
);



/**
 * @brief Encodes data using XOR-based erasure coding.
 * @param data_buffer Pointer to the data buffer.
 * @param parity_buffer Pointer to the parity buffer.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @return XorecResult XorecResult indicating success or failure.
 */
XorecResult xorec_unified_prefetch_encode(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  XorecVersion version
);

/**
 * @brief Decodes data using XOR-based erasure coding.
 * 
 * @param data_buffer Pointer to the data buffer.
 * @param parity_buffer Pointer to the parity buffer.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @param block_bitmap A bitmap indicating which blocks are present.
 * @return XorecResult XorecResult indicating success or failure.
 */
XorecResult xorec_unified_prefetch_decode(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap,
  XorecVersion version
);


/**
 * @brief Encodes data using XOR-based erasure coding.
 * 
 * @param gpu_data_buffer Pointer to the data buffer in GPU memory.
 * @param cpu_data_buffer Pointer to the data buffer in CPU memory (used for computation)
 * @param parity_buffer Pointer to the parity buffer.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @return XorecResult XorecResult indicating success or failure.
 */
XorecResult xorec_gpu_prefetch_encode(
  const uint8_t *XOREC_RESTRICT gpu_data_buffer,
  uint8_t *XOREC_RESTRICT cpu_data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  size_t prefetch_bytes,
  XorecVersion version
);

/**
 * @brief Decodes data using XOR-based erasure coding.
 * 
 * @param gpu_data_buffer Pointer to the data buffer in GPU memory.
 * @param cpu_data_buffer Pointer to the data buffer in CPU memory (used for computation)
 * @param parity_buffer Pointer to the parity buffer.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @param block_bitmap A bitmap indicating which blocks are present.
 * @return XorecResult XorecResult indicating success or failure.
 */
XorecResult xorec_gpu_prefetch_decode(
  uint8_t *XOREC_RESTRICT gpu_data_buffer,
  uint8_t *XOREC_RESTRICT cpu_data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap,
  size_t prefetch_bytes,
  XorecVersion version
);

/**
 * @brief XORs two blocks of data with AVX512 SIMD instructions.
 * 
 * @param dest Pointer to the destination block.
 * @param src Pointer to the source block.
 * @param bytes Number of bytes to XOR.
 */
static void inline xorec_xor_blocks_avx512(void * XOREC_RESTRICT dest, const void * XOREC_RESTRICT src, size_t bytes) {
  #if defined(TRY_XOREC_AVX512)
    XOREC_AVX512 * XOREC_RESTRICT dest512 = reinterpret_cast<XOREC_AVX512*>(dest);
    const XOREC_AVX512 * XOREC_RESTRICT src512 = reinterpret_cast<const XOREC_AVX512*>(src);

    for (; bytes >= 256; bytes -= 256, dest512 += 4, src512 += 4) {
      XOREC_AVX512 x0 = _mm512_xor_si512(_mm512_loadu_si512(dest512), _mm512_loadu_si512(src512));
      XOREC_AVX512 x1 = _mm512_xor_si512(_mm512_loadu_si512(dest512 + 1), _mm512_loadu_si512(src512 + 1));
      XOREC_AVX512 x2 = _mm512_xor_si512(_mm512_loadu_si512(dest512 + 2), _mm512_loadu_si512(src512 + 2));
      XOREC_AVX512 x3 = _mm512_xor_si512(_mm512_loadu_si512(dest512 + 3), _mm512_loadu_si512(src512 + 3));
      _mm512_storeu_si512(dest512, x0);
      _mm512_storeu_si512(dest512 + 1, x1);
      _mm512_storeu_si512(dest512 + 2, x2);
      _mm512_storeu_si512(dest512 + 3, x3);
    }

    if (bytes > 64) {
      XOREC_AVX512 x0 = _mm512_xor_si512(_mm512_loadu_si512(dest512), _mm512_loadu_si512(src512));
      XOREC_AVX512 x1 = _mm512_xor_si512(_mm512_loadu_si512(dest512 + 1), _mm512_loadu_si512(src512 + 1));
      _mm512_storeu_si512(dest512, x0);
      _mm512_storeu_si512(dest512 + 1, x1);
      dest512 += 2;
      src512 += 2;
    }

    if (bytes > 0) {
      XOREC_AVX512 x0 = _mm512_xor_si512(_mm512_loadu_si512(dest512), _mm512_loadu_si512(src512));
      _mm512_storeu_si512(dest512, x0);
    }
    
  #else
    throw_error("AVX512 not supported");
  #endif
}


/**
 * @brief XORs two blocks of data with AVX2 SIMD instructions.
 * 
 * @param dest Pointer to the destination block.
 * @param src Pointer to the source block.
 * @param bytes Number of bytes to XOR.
 */
static void inline xorec_xor_blocks_avx2(void * XOREC_RESTRICT dest, const void * XOREC_RESTRICT src, size_t bytes) {
  #if defined(TRY_XOREC_AVX2)
    XOREC_AVX2 * XOREC_RESTRICT dest256 = reinterpret_cast<XOREC_AVX2*>(dest);
    const XOREC_AVX2 * XOREC_RESTRICT src256 = reinterpret_cast<const XOREC_AVX2*>(src);

    for (; bytes >= 128; bytes -= 128, dest256 += 4, src256 += 4) {
      XOREC_AVX2 x0 = _mm256_xor_si256(_mm256_loadu_si256(dest256), _mm256_loadu_si256(src256));
      XOREC_AVX2 x1 = _mm256_xor_si256(_mm256_loadu_si256(dest256 + 1), _mm256_loadu_si256(src256 + 1));
      XOREC_AVX2 x2 = _mm256_xor_si256(_mm256_loadu_si256(dest256 + 2), _mm256_loadu_si256(src256 + 2));
      XOREC_AVX2 x3 = _mm256_xor_si256(_mm256_loadu_si256(dest256 + 3), _mm256_loadu_si256(src256 + 3));
      _mm256_storeu_si256(dest256, x0);
      _mm256_storeu_si256(dest256 + 1, x1);
      _mm256_storeu_si256(dest256 + 2, x2);
      _mm256_storeu_si256(dest256 + 3, x3);
    }

    if (bytes > 0) {
      XOREC_AVX2 x0 = _mm256_xor_si256(_mm256_loadu_si256(dest256), _mm256_loadu_si256(src256));
      XOREC_AVX2 x1 = _mm256_xor_si256(_mm256_loadu_si256(dest256 + 1), _mm256_loadu_si256(src256 + 1));
      _mm256_storeu_si256(dest256, x0);
      _mm256_storeu_si256(dest256 + 1, x1);
    }
  #else
    throw_error("AVX2 not supported");
  #endif
}


/**
 * @brief XORs two blocks of data with AVX SIMD instructions.
 * 
 * @param dest Pointer to the destination block.
 * @param src Pointer to the source block.
 * @param bytes Number of bytes to XOR.
 */
static void inline xorec_xor_blocks_avx(void * XOREC_RESTRICT dest, const void * XOREC_RESTRICT src, size_t bytes) {
  #if defined(TRY_XOREC_AVX)
    XOREC_AVX * XOREC_RESTRICT dest128 = reinterpret_cast<XOREC_AVX*>(dest);
    const XOREC_AVX * XOREC_RESTRICT src128 = reinterpret_cast<const XOREC_AVX*>(src);

    for (; bytes >= 64; bytes -= 64, dest128 += 4, src128 += 4) {
      XOREC_AVX x0 = _mm_xor_si128(_mm_loadu_si128(dest128), _mm_loadu_si128(src128));
      XOREC_AVX x1 = _mm_xor_si128(_mm_loadu_si128(dest128 + 1), _mm_loadu_si128(src128 + 1));
      XOREC_AVX x2 = _mm_xor_si128(_mm_loadu_si128(dest128 + 2), _mm_loadu_si128(src128 + 2));
      XOREC_AVX x3 = _mm_xor_si128(_mm_loadu_si128(dest128 + 3), _mm_loadu_si128(src128 + 3));
      _mm_storeu_si128(dest128, x0);
      _mm_storeu_si128(dest128 + 1, x1);
      _mm_storeu_si128(dest128 + 2, x2);
      _mm_storeu_si128(dest128 + 3, x3);
    }
  #else
    throw_error("AVX not supported");
  #endif
}


/**
 * @brief XORs two blocks of data with scalar instructions.
 * 
 * @attention This explicitly prevents vectorization to ensure scalar instructions are used.
 * 
 * @param dest Pointer to the destination block.
 * @param src Pointer to the source block.
 * @param bytes Number of bytes to XOR.
 */
#pragma GCC push_options
#pragma GCC optimize ("no-tree-vectorize")
static void inline xorec_xor_blocks_scalar(void * XOREC_RESTRICT dest, const void * XOREC_RESTRICT src, size_t bytes) {
  uint64_t * XOREC_RESTRICT dest64 = reinterpret_cast<uint64_t*>(dest);
  const uint64_t * XOREC_RESTRICT src64 = reinterpret_cast<const uint64_t*>(src);

  for (; bytes >= 32; bytes -= 32, dest64 += 4, src64 += 4) {
    *dest64 ^= *src64;
    *(dest64 + 1) ^= *(src64 + 1);
    *(dest64 + 2) ^= *(src64 + 2);
    *(dest64 + 3) ^= *(src64 + 3);
  }
}
#pragma GCC pop_options

#endif // XOREC_HPP