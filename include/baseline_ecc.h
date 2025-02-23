#ifndef BASELINE_ECC_H
#define BASELINE_ECC_H

#include <cstdint>
#include <vector>
/*
 * Custom ECC implementation, used as a baseline for comparison
 *
*/


struct Encoder {
  int Mpackets;
  int Rpackets;
  int Nsegs;
};

struct Decoder {
  int Mpackets;
  int Rpackets;
  int Nsegs;
  int Nextra; /// This is the number of extra packets needed to decode the message, which is equal
              /// to the number of message packets that were not received. Nextra can range between
              /// 0 and Rpackets (the total number of redundant packets sent on the encoding side).
};

void rs_init_tables();
  
Encoder rs_create_encoder(int Mpackets, int Rpackets, size_t packet_size);
Decoder rs_create_decoder(int Mpackets, int Rpackets, size_t packet_size, int num_packets_received);

void rs_encode(Encoder &enc, uint8_t *orig_data, uint8_t *redundant_data);
void rs_decode(Decoder &dec, const uint8_t *inp_recv_orig_data, const uint8_t *inp_recv_redund_data, uint8_t *outp_data, uint32_t *recv_idx, uint32_t *inv_mat);

void rs_invert_cauchy_matrix(uint32_t *inv_mat, int Nextra, std::vector<int> &RowInd, std::vector<int> &ColInd);

#endif // BASELINE_ECC_H