#ifndef BASELINE_ECC_H
#define BASELINE_ECC_H

#include <cstdint>
#include <vector>

#
/*
 * Custom ECC implementation, used as a baseline for comparison
 *
*/

// Constants
const int WSIZE = 32; ///< The word size we use
const int L = 8; ///< The field we use is GF(2^L) = GF(256)
const int GF256_MULT_FIELD = (1 << L) - 1; ///< Size of the multiplicative group of the finite field.
                                                ///< The Multiplicative group doesn't include 1 => |GF(2^L)|-1
int GF256_EXP_TABLE[512 * 2 + 1];
int GF256_LOG_TABLE[256];
int BIT_TABLE[32];
const int MAX_RPACKETS = 128; // We can have up to 128 rpackets and 128 mpackets TODO: make this better
const int MAX_MPACKETS = 128; // We can have up to 128 rpackets and 128 mpackets TODO: make this better



struct Baseline_Params {
  int Mpackets;    ///< Number of message packets sent (original data)
  int Rpackets;    ///< Number of redundant packets sent (recovery data)

  int Nsegs;       ///< Number of segments in each packet
                        ///< Since we use a wsize of 32 bits, the size of our packets
                        ///< Is 32 * L * Nsegs bits

  int *orig_data;   ///< Original data TODO: specify size constraints
  int *redundant_data; ///< Redundant data TODO: specify size constraints
};




Baseline_Params baseline_get_encode_params(
  uint32_t num_original_blocks,
  uint32_t num_recovery_blocks,
  uint32_t block_size,
  void *orig_data,
  void *redundant_data
);

void baseline_encode(Baseline_Params& params);

void baseline_decode(Baseline_Params& params, uint32_t num_lost_blocks, uint32_t *lost_block_idxs);




#endif // BASELINE_ECC_H