#ifndef BASELINE_ECC_H
#define BASELINE_ECC_H

#include <cstdint>
#include <vector>
/*
 * Custom ECC implementation, used as a baseline for comparison
 *
*/
const int L = 8;
const int MultField = (1 << L) - 1; // The size of the multiplicative group of the finite field. This is 2^L - 1
int ExpToFieldElt[MultField + 1];   // A table that goes from the exponent of an element, in
                                    // terms of a previously chosen generator of the
                                    // multiplicative group, to its representation as a
                                    // vector over GF[2]
int FieldEltToExp[MultField + 1];   // The table that goes from the vector representation of
                                    // an element to its exponent of the generator.
int Bit[L];                         // An array of integers that is used to select individual bits:
                                    // (A & Bit[i]) is equal to 1 if the i-th bit of A is 1, and 0
                                    // otherwise.

struct Encoder {
  int Mpackets;
  int Rpackets;
  int Nsegs;
};
  
Encoder rs_create_encoder(int Mpackets, int Rpackets, size_t packet_size);
void rs_encode(Encoder &enc, uint8_t *orig_data, uint8_t *redundant_data);

#endif // BASELINE_ECC_H