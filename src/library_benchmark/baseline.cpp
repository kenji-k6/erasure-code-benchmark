#include "baseline.h"
#include <iostream>
#include <cstring>

// Macro to check whether the i-th bit of x is set
#define BIT(x, i) (x & (1 << i))
// Macro to access the "Cauchy submatrix" as described in the paper
// TODO: Maybe adjust M macro
#define M(i, j) ((*redundant_packets)[RowInd[i]][j])


/// Constants
const uint16_t MAX_RPACKETS = 256;       ///< We can have up to 128 rpackets and 128 mpackets TODO: make this better
const uint16_t MAX_MPACKETS = 256;       ///< We can have up to 128 rpackets and 128 mpackets TODO: make this better
const uint32_t WSIZE = 32;               ///< The word size we use
const uint32_t Lfield = 8;                    ///< The field we use is GF(2^L) = GF(256)
const uint32_t MultField = (1 << Lfield) - 1; ///< Size of the multiplicative group of the finite field.
                                    ///< The Multiplicative group doesn't include 1 => |GF(2^L)|-1

const uint32_t GF256_POLYNOMIAL = 0x8e;        ///< Irreducible Polynomial for GF(256)

/// Global Tables
uint8_t ExpToFieldElt[512 * 2 + 1];     ///< A table that goes from the exponent of an element, in
                                    ///< terms of a previously chosen generator of the multiplicative
                                    ///< group, to its representation as a vector over GF[2]
                                    ///< todo: check if can be smaller

uint16_t FieldEltToExp[256];             ///< The table that goes from the vector representation of
                                    ///< an element to its exponent of the generator.


/// Buffers required for decoding, need to be zeroed before use
uint32_t RowInd[MAX_RPACKETS];  ///< An array that keeps track of the extra (redundant)
                                ///< packets received: RowInd[i] is the identifier (usually index)
                                ///< of the i-th extra packet that was received.

uint32_t ColInd[MAX_MPACKETS];  ///< An array that keeps track of the message (orig. data)
                                ///< packets received: ColInd[i] is the identifier (usually index)
                                ///< of the i-th message packet that was received.

bool RecIndex[MAX_MPACKETS];    // TODO: Initialize and document properly

// TODO: Add documentation
uint32_t C[MAX_RPACKETS];
uint32_t D[MAX_RPACKETS];
uint32_t E[MAX_RPACKETS];
uint32_t F[MAX_RPACKETS];




// TODO: Init the decoding stuff here aswell
void baseline_init() {
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
}



Baseline_Params baseline_get_params(uint32_t num_original_blocks, uint32_t num_recovery_blocks, uint32_t block_size, void *orig_data, void *redundant_data) {
  if (block_size % WSIZE) {
    std::cerr << "Block size (" << block_size << ") must be a multiple of " << WSIZE << ".\n";
    exit(0);
  }

  if (num_original_blocks > MAX_MPACKETS || num_original_blocks > MAX_RPACKETS || (num_original_blocks + num_recovery_blocks) > MAX_MPACKETS) {
    std::cerr << "Total blocks can't be more than " << MAX_MPACKETS << '\n';
  }

  Baseline_Params params;
  params.Mpackets = num_original_blocks;
  params.Rpackets = num_recovery_blocks;
  params.Nsegs = block_size / (4*Lfield);
  params.orig_data = orig_data;
  params.redundant_data = redundant_data; // Usually interpreted as a matrix of 32bit words. The matrix has Rpacket rows
                                          // and Nseg * L rows

  return params;
}




void baseline_encode(Baseline_Params& params) {
  uint32_t Mpackets = params.Mpackets;
  uint32_t Rpackets = params.Rpackets;
  uint32_t Nsegs = params.Nsegs;
  uint32_t *message = static_cast<uint32_t*>(params.orig_data);
  auto redundant_packets = static_cast<uint32_t (*)[Rpackets][Nsegs*Lfield]>(params.redundant_data);

  for (uint32_t row = 0; row < Rpackets; ++row) { ///< The number of rows in our Cauchy matrix
                                                  ///< is equal to the number of redundant packets.
    auto packet = (*redundant_packets)[row];

    for (uint32_t col = 0; col < params.Mpackets; ++col) {  ///< The number of columns in our Cauchy matrix is equal to
                                                            ///< the number of information (original data) packets.
      // TODO: Checked signed-ness
      uint32_t exponent = (MultField - FieldEltToExp[row ^ col ^ MultField]) % MultField; ///< exponent is the multiplicative exponent of the element
                                                                                          ///< of the Cauchy matrix we are currently looking at.
                                                                      
      for (uint32_t row_bit = 0; row_bit < Lfield; ++row_bit) { ///< Each element of our finite field is now represented
                                                                ///< as an Lfield * Lfield 0-1 matrix.
        uint32_t *local_packet = packet + row_bit*Nsegs;

        for (uint32_t col_bit = 0; col_bit < Lfield; ++col_bit) {

          if (BIT(ExpToFieldElt[exponent + row_bit], col_bit)) {
            uint32_t *local_message = message + (col_bit * Nsegs) + (col * Lfield * Nsegs);
            for (uint32_t segment = 0; segment < Nsegs; ++segment) {
              local_packet[segment] ^= local_message[segment];
            }
          }
        }
      }
    }
  }
}






/// DECODING STUFF BELOW
static void invert_cauchy_matrix(
  uint32_t Nextra,  ///< No. of extra packets needed to decode the message,
                    ///< which is equal to the no. of message (orig. data) packets
                    ///< that were not received.
                    ///< @attention 0 <= Nextra <= Rpackets must hold

  uint32_t *InvMatPtr  ///< Preallocated memory of sizeof(uint32_t) * Nextra * Nextra bytes
) {
  // cast invmat to allow for row/column accesses
  auto InvMat = reinterpret_cast<int32_t (*)[Nextra][Nextra]>(InvMatPtr);
  for (uint32_t row = 0; row < Nextra; ++row) {
    for (uint32_t col = 0; col < Nextra; ++col) {
      if (col != row) {
        C[row] += FieldEltToExp[ RowInd[row] ^ RowInd[col] ];
        D[col] += FieldEltToExp[ ColInd[row] ^ ColInd[col] ];
      }

      E[row] += FieldEltToExp[ RowInd[row] ^ ColInd[col] ^ MultField ];
      F[col] += FieldEltToExp[ RowInd[row] ^ ColInd[col] ^ MultField ];
    }
  }
  for (uint32_t row = 0; row < Nextra; ++row) {
    for (uint32_t col = 0; col < Nextra; ++col) {
      (*InvMat)[row][col] = E[col] + F[row] - C[col] - D[row] - FieldEltToExp[ RowInd[col] ^ ColInd[row] ^ MultField ];
      if ((*InvMat)[row][col] >= 0) {
        (*InvMat)[row][col] = (*InvMat)[row][col] % MultField;
      } else {
        (*InvMat)[row][col] = (MultField - ((-(*InvMat)[row][col]) % MultField)) % MultField;
      }
    }
  }
}


static void update_redundant_packets(
  Baseline_Params& params,
  uint32_t Nextra
) {
  uint32_t Mpackets = params.Mpackets;
  uint32_t Rpackets = params.Rpackets;
  uint32_t Nsegs = params.Nsegs;
  uint32_t *message = static_cast<uint32_t*>(params.orig_data);
  auto redundant_packets = static_cast<uint32_t (*)[Rpackets][Nsegs*Lfield]>(params.redundant_data);
  for (uint32_t row = 0; row < Nextra; ++row) {
    for (uint32_t col = 0; col < Mpackets; ++col) {
      if (RecIndex[col]) {
        uint32_t exponent = (MultField - FieldEltToExp[ RowInd[row] ^ col ^ MultField ]) % MultField;
        for (uint32_t row_bit = 0; row_bit < Lfield; ++row_bit) {
          for (uint32_t col_bit = 0; col_bit < Lfield; ++col_bit) {
            if (BIT(ExpToFieldElt[exponent + row_bit], col_bit)) {
              for (uint32_t segment = 0; segment < Nsegs; segment++) {
                uint32_t *packet = (*redundant_packets)[row];
                uint32_t *local_packet = packet + row_bit*Nsegs; // TODO: Check if we have to take RowInd here!!
                local_packet[segment] /* M(row_bit + row*Lfield, segment)*/ ^= message[segment + col_bit*Nsegs + col * Lfield * Nsegs];
              }
            }
          }
        }
      }
    }
  }
}


// Row Ind <-> Col Ind error
static void multiply_updated_redundant_packets(
  Baseline_Params& params,
  uint32_t Nextra,
  uint32_t *InvMatPtr
) {
  auto InvMat = reinterpret_cast<uint32_t (*)[Nextra][Nextra]>(InvMatPtr);
  uint32_t Nsegs = params.Nsegs;
  uint32_t *message = static_cast<uint32_t*>(params.orig_data);
  uint32_t Rpackets = params.Rpackets;
  auto redundant_packets = static_cast<uint32_t (*)[Rpackets][Nsegs*Lfield]>(params.redundant_data);

  for (int row = 0; row < Nextra; row++) {
    for (int col = 0; col < Nextra; col++) {
      uint32_t exponent = (*InvMat)[row][col];
      for (int row_bit = 0; row_bit < Lfield; row_bit++) {
        for (int col_bit = 0; col_bit < Lfield; col_bit++) {
          if (BIT(ExpToFieldElt[exponent + row_bit], col_bit)) {
            for (int segment = 0; segment < Nsegs; segment++) {
              uint32_t *packet = (*redundant_packets)[col];
              uint32_t *local_packet = packet + col_bit*Nsegs;
              message[(ColInd[row] * Lfield * Nsegs) + (row_bit * Nsegs) + segment] ^=
                local_packet[segment];
            }
          }
        }
      }
    }
  }
}

static void zero_decode_buffers() {
  memset(&RowInd, 0, MAX_RPACKETS*sizeof(uint32_t));
  memset(&ColInd, 0, MAX_MPACKETS*sizeof(uint32_t));
  memset(&RecIndex, 0, MAX_MPACKETS*sizeof(bool));
  memset(&C, 0, MAX_RPACKETS*sizeof(uint32_t));
  memset(&D, 0, MAX_RPACKETS*sizeof(uint32_t));
  memset(&E, 0, MAX_RPACKETS*sizeof(uint32_t));
  memset(&F, 0, MAX_RPACKETS*sizeof(uint32_t));
}

void baseline_decode(
  Baseline_Params& params,
  uint32_t *InvMatPtr,
  uint32_t num_lost_blocks,
  uint32_t *lost_block_idx
) {
  // Compute RowInd, ColInd, RecIndex
  uint32_t Mpackets = params.Mpackets;
  uint32_t Rpackets = params.Rpackets;

  uint32_t row_idx = 0;
  uint32_t col_idx = 0;
  uint32_t lost_arr_idx = 0;
  uint32_t Nextra = 0;
  uint32_t i = 0;

  if (num_lost_blocks > Rpackets) return;
  zero_decode_buffers();

  for (; i < Mpackets; ++i) {
    if (lost_arr_idx < num_lost_blocks && lost_block_idx[lost_arr_idx] == i) {
      //lost data block
      lost_arr_idx++;
      RecIndex[i] = false;
      ++Nextra;
      continue;
    }

    ColInd[col_idx++] = i;
    RecIndex[i] = true;
  }
  for (; i < Mpackets+Rpackets; ++i) {
    if (lost_arr_idx < num_lost_blocks && lost_block_idx[lost_arr_idx] == i) {
      //lost data block
      lost_arr_idx++;
      continue;
    }
    RowInd[row_idx++] = i; 
  }

  invert_cauchy_matrix(Nextra, InvMatPtr);
  update_redundant_packets(params, Nextra);
  std::cout << "hola1\n";
  multiply_updated_redundant_packets(params, Nextra, InvMatPtr);
}




