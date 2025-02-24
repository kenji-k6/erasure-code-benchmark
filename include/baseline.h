#ifndef BASELINE_H
#define BASELINE_H
#include <cstdint>


struct Baseline_Params {
  int Mpackets;    ///< Number of message packets sent (original data)
  int Rpackets;    ///< Number of redundant packets sent (recovery data)

  int Nsegs;       ///< Number of segments in each packet
                        ///< Since we use a wsize of 32 bits, the size of our packets
                        ///< Is 32 * L * Nsegs bits

  int *orig_data;   ///< Original data TODO: specify size constraints
  int *redundant_data; ///< Redundant data TODO: specify size constraints
};


void init_tables();

Baseline_Params baseline_get_params(
  uint32_t num_original_blocks,
  uint32_t num_recovery_blocks,
  uint32_t block_size,
  void *orig_data,
  void *redundant_data
);

#endif