#include "baseline_ecc.h"
#include <cstdlib>
#include <iostream>
#include <cstring>


// Constants / Global stuff
uint32_t Lfield = L;
uint32_t MultField = GF256_MULT_FIELD;
int *ExpToFieldElt = GF256_EXP_TABLE;
int *FieldEltToExp = GF256_LOG_TABLE;
int *Bit = BIT_TABLE;

// Macro to access the Cauchy Submatrix
#define M(i, j) redundant_data[(RowInd[i] * Lfield * Nsegs) + j * Nsegs] /* Might have to adjust macro */

Baseline_Params baseline_get_encode_params(uint32_t num_original_blocks, uint32_t num_recovery_blocks, uint32_t block_size, void *orig_data, void *redundant_data) {
  if (block_size % WSIZE != 0) {
    std::cerr << "Block Size (" << block_size << ") must be a multiple of " << Lfield << ".\n";
    exit(0);
  }

  Baseline_Params params;
  params.Mpackets = num_original_blocks;
  params.Rpackets = num_recovery_blocks;
  params.Nsegs = block_size / WSIZE;
  params.orig_data = (int*) orig_data;
  params.redundant_data = (int*) redundant_data;
  return params;
}

void baseline_encode(Baseline_Params& params) {
  int Mpackets = params.Mpackets;
  int Rpackets = params.Rpackets;
  int Nsegs = params.Nsegs;
  int *orig_data = params.orig_data;
  int *redundant_data = params.redundant_data;

  int col; ///< The number of columns in our Cauchy matrix
                ///< is equal to the number of information (original data) packets.
  for (int row = 0; row < Rpackets; row++) { ///< The number of rows in our Cauchy matrix
                                                  ///< is equal to the number of redundant packets.

    int *packet = redundant_data + (row * Lfield * Nsegs); //< We are computing the "row"-th redundant packet

    for (int col = 0; col < Mpackets; col++) { ///< The number of columns in our Cauchy matrix is equal to
                                                    ///< the number of information (original data) packets.

      int exponent = (MultField - FieldEltToExp[row ^ col ^ MultField]) % MultField; ///< exponent is the multiplicative exponent of the element
                                                                                          ///< of the Cauchy matrix we are currently looking at.

      for (int row_bit = 0; row_bit < Lfield; row_bit++) { ///< Each element of our finite field is now represented
                                                                ///< as an Lfield * Lfield 0-1 matrix.
        int *local_packet = packet + (row_bit * Nsegs);

        for (int col_bit = 0; col_bit < Lfield; col_bit++) {

          /// Check if the current bit of the matrix element is 1.
          if (ExpToFieldElt[exponent+row_bit] & Bit[col_bit]) {
            int *orig_packet = orig_data + (col * Lfield * Nsegs); // We are considering the "col"-th original packet
            int *local_orig_packet = orig_packet + (col_bit * Nsegs);

            for (int segment = 0; segment < Nsegs; segment++) {
              local_packet[segment] ^= local_orig_packet[segment];
            }
          }
        }
      }
    }
  }
}



struct decode_buffers_t {
  int RowInd[MAX_RPACKETS];       ///< An array that keeps track of the extra (redundant)
                                  ///< packets received: RowInd[i] is the identifier (usually index)
                                  ///< of the i-th extra packet that was received.

  int ColInd[MAX_MPACKETS];       ///< An array that keeps track of the message (orig. data)
                                  ///< packets received: ColInd[i] is the identifier (usually index)
                                  ///< of the i-th message packet that was received.

  bool RecIndex[MAX_MPACKETS];       // TODO: Initialize and document properly

  ///< Auxiliarry buffers that have to be 
  int C[MAX_RPACKETS];
  int D[MAX_RPACKETS];
  int E[MAX_RPACKETS];
  int F[MAX_RPACKETS];
} Decode_Buffers;

void invert_cauchy_matrix(
  int Nextra,       ///< No. of extra packets needed to decode the message,
                    ///< which is equal to the no. of message (orig. data) packets
                    ///< that were not received.
                    ///< @attention 0 <= Nextra <= Rpackets must hold
  int *InvMat       ///< Preallocated memory of sizeof(uint32_t) * Nextra * Nextra bytes
) {
  int *RowInd = Decode_Buffers.RowInd;
  int *ColInd = Decode_Buffers.ColInd;
  int *C = Decode_Buffers.C;
  int *D = Decode_Buffers.D;
  int *E = Decode_Buffers.E;
  int *F = Decode_Buffers.F;

  for (int row = 0; row < Nextra; row++) {
    for (int col = 0; col < Nextra; col++) {
      if (col != row) {
        C[row] += FieldEltToExp[ RowInd[row] ^ RowInd[col] ];
        D[col] += FieldEltToExp[ ColInd[row] ^ ColInd[col] ];
      }

      E[row] += FieldEltToExp[ RowInd[row] ^ ColInd[col] ^ MultField ];
      F[col] += FieldEltToExp[ RowInd[row] ^ ColInd[col] ^ MultField ];
    }
  }

  for (int row = 0; row < Nextra; row++) {
    for (int col = 0; col < Nextra; col++) {
      ///< InvMat[(row * Nextra) + col] = "InvMat[row][col]"
      InvMat[(row * Nextra) + col] = E[col] + F[row] - C[col] - D[row]
                                   - FieldEltToExp[ RowInd[col] ^ ColInd[row] ^ MultField ];
      if (InvMat[(row * Nextra) + col] >= 0) {
        InvMat[(row * Nextra) + col] = InvMat[(row * Nextra) + col] % MultField;
      } else {
        InvMat[(row * Nextra) + col] = (MultField - ((-InvMat[(row * Nextra) + col]) % MultField)) % MultField;
      }
    }
  }
}

void update_redundant_packets(
  Baseline_Params& params,
  int Nextra
) {
  int Mpackets = params.Mpackets;
  int Nsegs = params.Nsegs;
  int *RowInd = Decode_Buffers.RowInd;
  int *orig_data = params.orig_data;
  int *redundant_data = params.redundant_data;
  bool *RecIndex = Decode_Buffers.RecIndex;

  for (int row = 0; row < Nextra; row++) {
    for (int col = 0; col < Mpackets; col++) {
      if (RecIndex[col]) {
        auto exponent = (MultField - FieldEltToExp[ RowInd[row] ^ col ^ MultField ]) % MultField;

        for (int row_bit = 0; row_bit < Lfield; row_bit++) {
          for (int col_bit = 0; col_bit < Lfield; col_bit++) {
            if (ExpToFieldElt[exponent+row_bit] & Bit[col_bit]) {
              for (int segment = 0; segment < Nsegs; segment++) {
                M(row_bit + row*Lfield, segment) ^= orig_data[(col*Lfield*Nsegs) + (col_bit * Nsegs) + segment];
              }
            }
          }
        }
      }
    }
  }
}

void multiply_updated_redundant_packets(
  Baseline_Params& params,
  int Nextra,
  int *InvMat
) {
  int Nsegs = params.Nsegs;
  int *orig_data = params.orig_data;
  int *redundant_data = params.redundant_data;
  int *ColInd = Decode_Buffers.ColInd;
  int *RowInd = Decode_Buffers.RowInd;
  for (int row = 0; row < Nextra; row++) {
    for (int col = 0; col < Nextra; col++) {
      int exponent = InvMat[(row * Nextra) + col];
      for (int row_bit = 0; row_bit < Lfield; row_bit++) {
        for (int col_bit = 0; col_bit < Lfield; col_bit++) {
          if (ExpToFieldElt[exponent + row_bit] & Bit[col_bit]) {
            for (int segment = 0; segment < Nsegs; segment++) {
              orig_data[(ColInd[row] * Lfield * Nsegs) + (row_bit * Nsegs + segment)] ^=
                M(col_bit + col*Lfield, segment);
            }
          }
        }
      }
    }
  }
}

void init_decode_buffers(Baseline_Params& params, uint32_t num_lost_blocks, uint32_t *lost_block_idxs) {
  memset(&Decode_Buffers.RowInd, 0, sizeof(int)*MAX_RPACKETS);
  memset(&Decode_Buffers.ColInd, 0, sizeof(int)*MAX_MPACKETS);
  memset(&Decode_Buffers.RecIndex, 0, sizeof(int)*MAX_MPACKETS);
  memset(&Decode_Buffers.C, 0, sizeof(int)*MAX_RPACKETS);
  memset(&Decode_Buffers.D, 0, sizeof(int)*MAX_RPACKETS);
  memset(&Decode_Buffers.E, 0, sizeof(int)*MAX_RPACKETS);
  memset(&Decode_Buffers.F, 0, sizeof(int)*MAX_RPACKETS);

  int lost_arr_idx = 0;
  int recv_orig_idx = 0;
  int recv_red_idx = 0;
  int i;


  for (i = 0; i < params.Mpackets; i++) {
    if (lost_arr_idx < num_lost_blocks && lost_block_idxs[lost_arr_idx] == i) { // lost packet
      lost_arr_idx++;
      continue;
    }

    // packet arrived
    Decode_Buffers.ColInd[recv_orig_idx++] = i;
  }

  for (; i < params.Mpackets+params.Rpackets; i++) {
    if (lost_arr_idx < num_lost_blocks && lost_block_idxs[lost_arr_idx] == i) {
      lost_arr_idx++;
      continue;
    }
    Decode_Buffers.RowInd[recv_red_idx++] = i; // TODO: Maybe adjust this to i - Mpackets
  }
}

void baseline_decode(Baseline_Params& params, uint32_t num_lost_blocks, uint32_t *lost_block_idxs, void *InvMat) {
  // TODO: Initialize Nextra
  int Nextra = 0;

  if (Nextra <= 0 || Nextra > MAX_RPACKETS) {
    return;
  }

  init_decode_buffers(params, num_lost_blocks, lost_block_idxs);
  invert_cauchy_matrix(Nextra, static_cast<int*>(InvMat));
  update_redundant_packets(params, Nextra);
  multiply_updated_redundant_packets(params, Nextra, static_cast<int*>(InvMat));
}

