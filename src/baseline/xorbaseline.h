#ifndef XORBASELINE_H
#define XORBASELINE_H

#include <bitset>
#include <cstdint>
#include <cstring>
#include <array>

/**
 * @file xorbaseline.h
 * @brief Provides encoding and decoding functions for custom XOR-based erasure coding.
 * 
 * This header defines the XOR-based erasure encoding and decoding functions,
 * optimized with SIMD intrinsics when available. It supports AVX and AVX2
 */


#define XOR_RESTRICT __restrict

#if defined(__AVX2__)
  #define TRY_XOR_AVX2
  #include <immintrin.h>
  #define XOR_AVX2 __m256i
#endif
#if defined(__AVX__)
  #define TRY_XOR_AVX
  #include <immintrin.h>
  #define XOR_AVX __m128i
#endif


constexpr uint32_t XOR_BLOCK_SIZE_MULTIPLE = 64;

constexpr uint32_t XOR_PTR_ALIGNMENT = 32;
constexpr uint32_t XOR_MIN_BLOCK_SIZE = 64;

constexpr uint32_t XOR_MIN_DATA_BLOCKS = 1;
constexpr uint32_t XOR_MAX_DATA_BLOCKS = 128;
constexpr uint32_t XOR_MIN_PARITY_BLOCKS = 1;
constexpr uint32_t XOR_MAX_PARITY_BLOCKS = 128;
constexpr uint32_t XOR_MAX_TOTAL_BLOCKS = 256;


/// Bitmap to check if all data blocks are available (no recovery needed)
const std::bitset<XOR_MAX_TOTAL_BLOCKS> COMPLETE_DATA_BITMAP =(
  std::bitset<256>(0xFFFFFFFFFFFFFFFFULL)<<64) |
  std::bitset<256>(0xFFFFFFFFFFFFFFFFULL
  );


/**
 * @enum XORResult
 * @brief Represents the result status of encoding and decoding operations.
 */
enum class XORResult {
  Success = 0,
  InvalidSize = 1,
  InvalidCounts = 2,
  InvalidAlignment = 3,
  DecodeFailure = 4
};

/**
 * @enum XORVersion
 * @brief Allows to specify which version of the implementations to use
 */
enum class XORVersion {
  Auto = 0,
  Scalar = 1,
  ScalarNoOpt = 2,
  AVX = 3,
  AVX2 = 4
};


/**
 * @brief Encodes data using XOR-based erasure coding.
 * @param data_buffer Pointer to the data buffer.
 * @param parity_buffer Pointer to the parity buffer.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @return XORResult XORResult indicating success or failure.
 */
XORResult xor_encode(
  const uint8_t *XOR_RESTRICT data_buffer,
  uint8_t *XOR_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  XORVersion version = XORVersion::Auto
);


/**
 * @brief Decodes data using XOR-based erasure coding.
 * @param data_buffer Pointer to the data buffer.
 * @param parity_buffer Pointer to the parity buffer.
 * @param block_size Size of each block in bytes.
 * @param num_data_blocks Number of data blocks.
 * @param num_parity_blocks Number of parity blocks.
 * @param block_bitmap A bitset indicating which blocks are present.
 * @return XORResult XORResult indicating success or failure.
 */
XORResult xor_decode(
  uint8_t *XOR_RESTRICT data_buffer,
  const uint8_t *XOR_RESTRICT parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  std::bitset<256> block_bitmap,   ///< Indexing for parity blocks starts at bit 128, e.g. the j-th parity block is at bit 128 + j, j < 128
  XORVersion version = XORVersion::Auto
);

static void inline XOR_xor_blocks_avx2(void * XOR_RESTRICT dest, const void * XOR_RESTRICT src, uint32_t bytes) {
  #if defined(TRY_XOR_AVX2)
    XOR_AVX2 * XOR_RESTRICT dest256 = reinterpret_cast<XOR_AVX2*>(dest);
    const XOR_AVX2 * XOR_RESTRICT src256 = reinterpret_cast<const XOR_AVX2*>(src);

    for (; bytes >= 128; bytes -= 128, dest256 += 4, src256 += 4) {
      XOR_AVX2 x0 = _mm256_xor_si256(_mm256_loadu_si256(dest256), _mm256_loadu_si256(src256));
      XOR_AVX2 x1 = _mm256_xor_si256(_mm256_loadu_si256(dest256 + 1), _mm256_loadu_si256(src256 + 1));
      XOR_AVX2 x2 = _mm256_xor_si256(_mm256_loadu_si256(dest256 + 2), _mm256_loadu_si256(src256 + 2));
      XOR_AVX2 x3 = _mm256_xor_si256(_mm256_loadu_si256(dest256 + 3), _mm256_loadu_si256(src256 + 3));
      _mm256_storeu_si256(dest256, x0);
      _mm256_storeu_si256(dest256 + 1, x1);
      _mm256_storeu_si256(dest256 + 2, x2);
      _mm256_storeu_si256(dest256 + 3, x3);
    }

    if (bytes > 0) {
      XOR_AVX2 x0 = _mm256_xor_si256(_mm256_loadu_si256(dest256), _mm256_loadu_si256(src256));
      XOR_AVX2 x1 = _mm256_xor_si256(_mm256_loadu_si256(dest256 + 1), _mm256_loadu_si256(src256 + 1));
      _mm256_storeu_si256(dest256, x0);
      _mm256_storeu_si256(dest256 + 1, x1);
    }
  #else
    std::cerr << "AVX2 not supported\n";
    exit(1);
  #endif
}

static void inline XOR_xor_blocks_avx(void * XOR_RESTRICT dest, const void * XOR_RESTRICT src, uint32_t bytes) {
  #if defined(TRY_XOR_AVX)
    XOR_AVX * XOR_RESTRICT dest128 = reinterpret_cast<XOR_AVX*>(dest);
    const XOR_AVX * XOR_RESTRICT src128 = reinterpret_cast<const XOR_AVX*>(src);

    for (; bytes >= 64; bytes -= 64, dest128 += 4, src128 += 4) {
      XOR_AVX x0 = _mm_xor_si128(_mm_loadu_si128(dest128), _mm_loadu_si128(src128));
      XOR_AVX x1 = _mm_xor_si128(_mm_loadu_si128(dest128 + 1), _mm_loadu_si128(src128 + 1));
      XOR_AVX x2 = _mm_xor_si128(_mm_loadu_si128(dest128 + 2), _mm_loadu_si128(src128 + 2));
      XOR_AVX x3 = _mm_xor_si128(_mm_loadu_si128(dest128 + 3), _mm_loadu_si128(src128 + 3));
      _mm_storeu_si128(dest128, x0);
      _mm_storeu_si128(dest128 + 1, x1);
      _mm_storeu_si128(dest128 + 2, x2);
      _mm_storeu_si128(dest128 + 3, x3);
    }
  #else
    std::cerr << "AVX not supported\n";
    exit(1);
  #endif
}

static void inline XOR_xor_blocks_scalar(void * XOR_RESTRICT dest, const void * XOR_RESTRICT src, uint32_t bytes) {
  uint64_t * XOR_RESTRICT dest64 = reinterpret_cast<uint64_t*>(dest);
  const uint64_t * XOR_RESTRICT src64 = reinterpret_cast<const uint64_t*>(src);
  #pragma aligned
  for (; bytes >= 32; bytes -= 32, dest64 += 4, src64 += 4) {
    *dest64 ^= *src64;
    *(dest64 + 1) ^= *(src64 + 1);
    *(dest64 + 2) ^= *(src64 + 2);
    *(dest64 + 3) ^= *(src64 + 3);
  }
}

static void inline XOR_xor_blocks_scalar_no_opt(void * XOR_RESTRICT dest, const void * XOR_RESTRICT src, uint32_t bytes) {
  uint64_t * XOR_RESTRICT dest64 = reinterpret_cast<uint64_t*>(dest);
  const uint64_t * XOR_RESTRICT src64 = reinterpret_cast<const uint64_t*>(src);

  #pragma novector
  for (; bytes >= 32; bytes -= 32, dest64 += 4, src64 += 4) {
    *dest64 ^= *src64;
    *(dest64 + 1) ^= *(src64 + 1);
    *(dest64 + 2) ^= *(src64 + 2);
    *(dest64 + 3) ^= *(src64 + 3);
  }
}


static void inline XOR_copy_blocks_avx2(void * XOR_RESTRICT dest, const void * XOR_RESTRICT src, uint32_t bytes) {
  #if defined(TRY_XOR_AVX2)
    XOR_AVX2 * XOR_RESTRICT dest256 = reinterpret_cast<XOR_AVX2*>(dest);
    const XOR_AVX2 * XOR_RESTRICT src256 = reinterpret_cast<const XOR_AVX2*>(src);

    for (; bytes >= 128; bytes -= 128, dest256 += 4, src256 += 4) {
      _mm256_storeu_si256(dest256, _mm256_loadu_si256(src256));
      _mm256_storeu_si256(dest256 + 1, _mm256_loadu_si256(src256 + 1));
      _mm256_storeu_si256(dest256 + 2, _mm256_loadu_si256(src256 + 2));
      _mm256_storeu_si256(dest256 + 3, _mm256_loadu_si256(src256 + 3));
    }

    if (bytes > 0) {
      _mm256_storeu_si256(dest256, _mm256_loadu_si256(src256));
      _mm256_storeu_si256(dest256 + 1, _mm256_loadu_si256(src256 + 1));
    }
  #else
    std::cerr << "AVX2 not supported\n";
    exit(1);
  #endif
}

static void inline XOR_copy_blocks_avx(void * XOR_RESTRICT dest, const void * XOR_RESTRICT src, uint32_t bytes) {
  #if defined(TRY_XOR_AVX)
    XOR_AVX * XOR_RESTRICT dest128 = reinterpret_cast<XOR_AVX*>(dest);
    const XOR_AVX * XOR_RESTRICT src128 = reinterpret_cast<const XOR_AVX*>(src);

    for (; bytes >= 64; bytes -= 64, dest128 += 4, src128 += 4) {
      _mm_storeu_si128(dest128, _mm_loadu_si128(src128));
      _mm_storeu_si128(dest128 + 1, _mm_loadu_si128(src128 + 1));
      _mm_storeu_si128(dest128 + 2, _mm_loadu_si128(src128 + 2));
      _mm_storeu_si128(dest128 + 3, _mm_loadu_si128(src128 + 3));
    }
  #else
    std::cerr << "AVX2 not supported\n";
    exit(1);
  #endif
}

static void inline XOR_copy_blocks_scalar(void * XOR_RESTRICT dest, const void * XOR_RESTRICT src, uint32_t bytes) { memcpy(dest, src, bytes); }


/**
 * @brief XORs two memory blocks.
 * @param dest Pointer to the destination block.
 * @param src Pointer to the source block.
 * @param bytes Number of bytes to XOR.
 */
static void inline XOR_xor_blocks(
  void * XOR_RESTRICT dest,
  const void * XOR_RESTRICT src,
  uint32_t bytes
) {
  #if defined(TRY_XOR_AVX2)
    XOR_xor_blocks_avx2(dest, src, bytes);
  #elif defined(TRY_XOR_AVX)
    XOR_xor_blocks_avx(dest, src, bytes);
  #else
    XOR_xor_blocks_scalar(dest, src, bytes);
  #endif
}


/**
 * @brief Copies one memory block to the next one.
 * @param dest Pointer to the destination block.
 * @param src Pointer to the source block.
 * @param bytes Number of bytes to copy.
 */
static void inline XOR_copy_blocks(
  void * XOR_RESTRICT dest,
  const void * XOR_RESTRICT src,
  uint32_t bytes
) {
  #if defined(TRY_XOR_AVX2)
    XOR_copy_blocks_avx2(dest, src, bytes);
  #elif defined(TRY_XOR_AVX)
    XOR_copy_blocks_avx(dest, src, bytes);
  #else
    XOR_copy_blocks_scalar(dest, src, bytes);
  #endif
}

#endif // XORBASELINE_H