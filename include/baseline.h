#ifndef BASELINE_H
#define BASELINE_H
#include <cstdint>


struct Baseline_Params {
  uint32_t Mpackets;    ///< Number of message packets sent (original data)
  uint32_t Rpackets;    ///< Number of redundant packets sent (recovery data)
  
  uint32_t Nsegs;       ///< Number of segments in each packet
                        ///< Since we use a wsize of 32 bits, the size of our packets
                        ///< Is 32 * L * Nsegs bits

  void *orig_data;   ///< Original data TODO: specify size constraints
  void *redundant_data; ///< Redundant data TODO: specify size constraints
};


void baseline_init();


Baseline_Params baseline_get_params(
  uint32_t num_original_blocks,
  uint32_t num_recovery_blocks,
  uint32_t block_size,
  void *orig_data,
  void *redundant_data
);


void baseline_encode(Baseline_Params& params);
void baseline_decode(
  Baseline_Params& params,
  uint32_t *InvMatPtr,
  uint32_t num_lost_blocks,
  uint32_t *lost_block_idx
);
#endif