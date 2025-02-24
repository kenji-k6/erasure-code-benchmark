#include "baseline.h"

/// Constants
const uint8_t MAX_RPACKETS = 128;       ///< We can have up to 128 rpackets and 128 mpackets TODO: make this better
const uint8_t MAX_MPACKETS = 128;       ///< We can have up to 128 rpackets and 128 mpackets TODO: make this better
const uint8_t WSIZE = 32;               ///< The word size we use
const uint8_t L = 8;                    ///< The field we use is GF(2^L) = GF(256)
const uint16_t MultField = (1 << L) - 1; ///< Size of the multiplicative group of the finite field.
                                    ///< The Multiplicative group doesn't include 1 => |GF(2^L)|-1

const uint32_t GF256_POLYNOMIAL = 0x8e;        ///< Irreducible Polynomial for GF(256)

/// Global Tables
uint8_t ExpToFieldElt[512 * 2 + 1];     ///< A table that goes from the exponent of an element, in
                                    ///< terms of a previously chosen generator of the multiplicative
                                    ///< group, to its representation as a vector over GF[2]
                                    ///< todo: check if can be smaller

uint16_t FieldEltToExp[256];             ///< The table that goes from the vector representation of
                                    ///< an element to its exponent of the generator.

uint32_t Bit[32];                        ///< An array of integers that is used to select individual bits:
                                    ///< (A & Bit[i]) is equal to 1 if the i-th bit of A is 1, and 0 otherwise.


void init_tables() {
  // Initialze exp and log tables
  FieldEltToExp[0] = 512;
  ExpToFieldElt[0] = 1;
  for (unsigned i = 1; i < MultField; ++i) {
    unsigned next = static_cast<unsigned>(ExpToFieldElt[i-1]) * 2;

    if (next > MultField) next ^= GF256_POLYNOMIAL;

    ExpToFieldElt[i] = static_cast<uint8_t>(next);
    FieldEltToExp[ExpToFieldElt[i]] = static_cast<uint16_t>(i);
  }
  ExpToFieldElt[255] = ExpToFieldElt[0];
  FieldEltToExp[ExpToFieldElt[255]] = 255;
  for (unsigned i = 256; i < 2 * 255; ++i) {
    ExpToFieldElt[i] = ExpToFieldElt[i % 255];
  }
  ExpToFieldElt[2 * 255] = 1;
  for (unsigned i = 2 * 255 + 1; i < 4 * 255; ++i) {
    ExpToFieldElt[i] = 0;
  }

  // Initialize Bit table
  for (unsigned i = 0; i < 32; i++) {
    Bit[i] = 1 << i;
  }
}

Baseline_Params baseline_get_params(
  uint32_t num_original_blocks,
  uint32_t num_recovery_blocks,
  uint32_t block_size,
  void *orig_data,
  void *redundant_data
);


