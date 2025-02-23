#include "baseline_ecc.h"



Encoder rs_create_encoder(int Mpackets, int Rpackets, size_t packet_size) {
  Encoder encoder;
  encoder.Mpackets = Mpackets;
  encoder.Rpackets = Rpackets;
  encoder.Nsegs = packet_size / L;
  return encoder;
}