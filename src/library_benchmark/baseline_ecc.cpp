#include "baseline_ecc.h"



Encoder rs_create_encoder(int Mpackets, int Rpackets, size_t packet_size) {
  Encoder encoder;
  encoder.Mpackets = Mpackets;
  encoder.Rpackets = Rpackets;
  encoder.Nsegs = packet_size / L;
  return encoder;
}


Decoder rs_create_decoder(int Mpackets, int Rpackets, size_t packet_size, int num_orig_packets_received) {
  Decoder decoder;
  decoder.Mpackets = Mpackets;
  decoder.Rpackets = Rpackets;
  decoder.Nsegs = packet_size / L;
  decoder.Nextra = Mpackets - num_orig_packets_received;
  return decoder;
}


void rs_encode(Encoder &enc, uint8_t *orig_data, uint8_t *redundant_data) {
  /// The number of rows in our Cauchy matrix is equal to the number of redundant packets
  for (int row = 0; row < enc.Rpackets; row++) {
    uint8_t *packet = redundant_data + row * enc.Nsegs * L;

    /// The number of columns in our Cauchy matrix is equal to the number of informaiton packets
    for (int col = 0; col < enc.Mpackets; col++) {
      /// exponent is the multiplicative exponent of the element of the Cauchy matrix
      int exponent = (MultField - FieldEltToExp[row ^ col ^ MultField]) % MultField;
      
      /// Each element of our finite field is now represented as a Lfield by Lfield 0-1 matrix.
      for (int row_bit = 0; row_bit < L; row_bit++) {
        uint8_t *local_packet = packet + row_bit * enc.Nsegs;
        
        for (int col_bit = 0; col_bit < L; col_bit++) {
          if (ExpToFieldElt[exponent + row_bit] & Bit[col_bit]) {
            uint8_t *local_orig_data = orig_data + col * enc.Nsegs * L + col_bit * enc.Nsegs;

            for (int segment = 0; segment < enc.Nsegs; segment++) {
              local_packet[segment] ^= local_orig_data[segment];
            }
          }
        }
      }
    }
  }
}