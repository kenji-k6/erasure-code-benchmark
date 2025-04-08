#ifndef XOREC_HPP
#define XOREC_HPP

#include "xorec_utils.hpp"
#include "utils.hpp"

#include <immintrin.h>
#define XOREC_AVX __m128i
#define XOREC_AVX2 __m256i
#define XOREC_AVX512 __m512i


void xorec_init(size_t num_data_blocks, size_t num_parity_blocks);

XorecResult xorec_encode_avx(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
);

XorecResult xorec_decode_avx(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap
);


XorecResult xorec_encode_avx2(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
);

XorecResult xorec_decode_avx2(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap
);

XorecResult xorec_encode_avx512(
  const uint8_t *XOREC_RESTRICT data_buffer,
  uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks
);

XorecResult xorec_decode_avx512(
  uint8_t *XOREC_RESTRICT data_buffer,
  const uint8_t *XOREC_RESTRICT parity_buffer,
  size_t block_size,
  size_t num_data_blocks,
  size_t num_parity_blocks,
  const uint8_t * XOREC_RESTRICT block_bitmap
);


static void inline xorec_xor_blocks_avx(void * XOREC_RESTRICT dest, const void * XOREC_RESTRICT src, size_t bytes) {
  XOREC_AVX * XOREC_RESTRICT dest128 = reinterpret_cast<XOREC_AVX*>(__builtin_assume_aligned(dest, 64));
  const XOREC_AVX * XOREC_RESTRICT src128 = reinterpret_cast<const XOREC_AVX*>(__builtin_assume_aligned(src, 64));
  for (; bytes >= 64; bytes -= 64, dest128 += 4, src128 += 4) {
    XOREC_AVX x0 = _mm_xor_si128(_mm_load_si128(dest128), _mm_load_si128(src128));
    XOREC_AVX x1 = _mm_xor_si128(_mm_load_si128(dest128 + 1), _mm_load_si128(src128 + 1));
    XOREC_AVX x2 = _mm_xor_si128(_mm_load_si128(dest128 + 2), _mm_load_si128(src128 + 2));
    XOREC_AVX x3 = _mm_xor_si128(_mm_load_si128(dest128 + 3), _mm_load_si128(src128 + 3));
    _mm_store_si128(dest128, x0);
    _mm_store_si128(dest128 + 1, x1);
    _mm_store_si128(dest128 + 2, x2);
    _mm_store_si128(dest128 + 3, x3);
  }
}


static void inline xorec_xor_blocks_avx2(void * XOREC_RESTRICT dest, const void * XOREC_RESTRICT src, size_t bytes) {
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


static void inline xorec_xor_blocks_avx512(void * XOREC_RESTRICT dest, const void * XOREC_RESTRICT src, size_t bytes) {
  XOREC_AVX512 * XOREC_RESTRICT dest512 = reinterpret_cast<XOREC_AVX512*>(__builtin_assume_aligned(dest, 64));
  const XOREC_AVX512 * XOREC_RESTRICT src512 = reinterpret_cast<const XOREC_AVX512*>(__builtin_assume_aligned(src, 64));
  for (; bytes >= 256; bytes -= 256, dest512 += 4, src512 += 4) {
    XOREC_AVX512 x0 = _mm512_xor_si512(_mm512_load_si512(dest512), _mm512_load_si512(src512));
    XOREC_AVX512 x1 = _mm512_xor_si512(_mm512_load_si512(dest512 + 1), _mm512_load_si512(src512 + 1));
    XOREC_AVX512 x2 = _mm512_xor_si512(_mm512_load_si512(dest512 + 2), _mm512_load_si512(src512 + 2));
    XOREC_AVX512 x3 = _mm512_xor_si512(_mm512_load_si512(dest512 + 3), _mm512_load_si512(src512 + 3));
    _mm512_store_si512(dest512, x0);
    _mm512_store_si512(dest512 + 1, x1);
    _mm512_store_si512(dest512 + 2, x2);
    _mm512_store_si512(dest512 + 3, x3);
  }
}


#endif // XOREC_HPP