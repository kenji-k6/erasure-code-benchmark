#ifndef BASELINE_ECC_H
#define BASELINE_ECC_H

#include <cstdint>
#include <vector>
/*
 * Custom ECC implementation, used as a baseline for comparison
 *
*/

// Constants
const uint32_t L = 8; ///< The field we use is GF(2^L) = GF(256)
const uint32_t GF256_MULT_FIELD = (1 << L) - 1; ///< Size of the multiplicative group of the finite field.
                                                ///< The Multiplicative group doesn't include 1 => |GF(2^L)|-1
uint32_t GF256_EXP_TABLE[512 * 2 + 1];
uint32_t GF256_LOG_TABLE[256];
uint32_t BIT_TABLE[32];

struct RS_encoder_params {
  uint32_t Mpackets;    ///< Number of message packets sent (original data)
  uint32_t Rpackets;    ///< Number of redundant packets sent (recovery data)
  uint32_t Nsegs;       ///< Number of segments in each packet
  uint8_t *orig_data;   ///< Original data TODO: specify size constraints
  uint8_t *redundant_data; ///< Redundant data TODO: specify size constraints
};

struct RS_decoder_params {
  size_t Mpackets;
  size_t Rpackets;
};



#endif // BASELINE_ECC_H