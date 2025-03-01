#ifndef XORBASELINE_H
#define XORBASELINE_H
#include <bitset>
#include <cstdint>

#if defined(__AVX512F__):
  #define XORBASELINE_AVX512
  #include <immintrin.h>
  constexpr uint32_t XORBASELINE_BLOCK_SIZE_MULTIPLE = 64;
#elif defined(__AVX2__):
  #define XORBASELINE_AVX2
  #include <immintrin.h>
  constexpr uint32_t XORBASELINE_BLOCK_SIZE_MULTIPLE = 32;
#elif defined(__AVX__):
  #define XORBASELINE_AVX
  #include <immintrin.h>
  constexpr uint32_t XORBASELINE_BLOCK_SIZE_MULTIPLE = 16;
#else
  constexpr uint32_t XORBASELINE_BLOCK_SIZE_MULTIPLE = 8;
#endif


constexpr uint32_t XORBASELINE_PTR_ALIGNMENT = 32;
constexpr uint32_t XORBASELINE_MIN_BLOCK_SIZE = 64;

constexpr uint32_t XORBASELINE_MIN_DATA_BLOCKS = 1;
constexpr uint32_t XORBASELINE_MAX_DATA_BLOCKS = 255;

constexpr uint32_t XORBASELINE_MIN_PARITY_BLOCKS = 1;
constexpr uint32_t XORBASELINE_MAX_PARITY_BLOCKS = 128;

constexpr uint32_t XORBASELINE_MAX_TOTAL_BLOCKS = 256;


typedef enum XORBaselineResult_t {
  XORBaseline_Success = 0,

  XORBaseline_InvalidSize = 1,
  
  XORBaseline_InvalidCounts = 2,

  XORBaseline_InvalidAlignment = 3
} XORBaselineResult;

XORBaselineResult encode(
  uint8_t *data_buffer,
  uint8_t *parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks
);

XORBaselineResult decode(
  uint8_t *data_buffer,
  uint8_t *parity_buffer,
  uint32_t block_size,
  uint32_t num_data_blocks,
  uint32_t num_parity_blocks,
  std::bitset<256> &lost_blocks
 );

#endif // XORBASELINE_H