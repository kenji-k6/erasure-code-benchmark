#include "baseline_ecc.h"
#include <cstdlib>
#include <iostream>
#include <cstring>


// Constants / Global stuff
uint32_t Lfield = L;
uint32_t MultField = GF256_MULT_FIELD;
uint32_t *ExpToFieldElt = GF256_EXP_TABLE;
uint32_t *FieldEltToExp = GF256_LOG_TABLE;
uint32_t *Bit = BIT_TABLE;

Baseline_Encoder_Params baseline_get_encode_params(uint32_t num_original_blocks, uint32_t num_recovery_blocks, uint32_t block_size) {
  if (block_size % Lfield != 0) {
    std::cerr << "Block Size (" << block_size << ") must be a multiple of " << Lfield << ".\n";
    exit(0);
  }

  Baseline_Encoder_Params params;
  params.Mpackets = num_original_blocks;
  params.Rpackets = num_recovery_blocks;
  params.Nsegs = block_size / Lfield;
  return params;
}

void baseline_encode(Baseline_Encoder_Params params) {
  uint32_t Mpackets = params.Mpackets;
  uint32_t Rpackets = params.Rpackets;
  uint32_t Nsegs = params.Nsegs;
  uint8_t *orig_data = params.orig_data;
  uint8_t *redundant_data = params.redundant_data;

  uint32_t col; ///< The number of columns in our Cauchy matrix
                ///< is equal to the number of information (original data) packets.
  for (uint32_t row = 0; row < Rpackets; row++) { ///< The number of rows in our Cauchy matrix
                                                  ///< is equal to the number of redundant packets.

    uint8_t *packet = redundant_data + (row * Lfield * Nsegs); //< We are computing the "row"-th redundant packet

    for (uint32_t col = 0; col < Mpackets; col++) { ///< The number of columns in our Cauchy matrix is equal to
                                                    ///< the number of information (original data) packets.

      uint32_t exponent = (MultField - FieldEltToExp[row ^ col ^ MultField]) % MultField; ///< exponent is the multiplicative exponent of the element
                                                                                          ///< of the Cauchy matrix we are currently looking at.

      for (uint32_t row_bit = 0; row_bit < Lfield; row_bit++) { ///< Each element of our finite field is now represented
                                                                ///< as an Lfield * Lfield 0-1 matrix.
        uint8_t *local_packet = packet + (row_bit * Nsegs);

        for (uint32_t col_bit = 0; col_bit < Lfield; col_bit++) {

          /// Check if the current bit of the matrix element is 1.
          if (ExpToFieldElt[exponent+row_bit] & Bit[col_bit]) {
            uint8_t *orig_packet = orig_data + (col * Lfield * Nsegs); // We are considering the "col"-th original packet
            uint8_t *local_orig_packet = orig_packet + (col_bit * Nsegs);

            for (uint32_t segment = 0; segment < Nsegs; segment++) {
              local_packet[segment] ^= local_orig_packet[segment];
            }
          }
        }
      }
    }
  }
}



// const int L = 8;
// const int MultField = (1 << L) - 1; // The size of the multiplicative group of the finite field. This is 2^L - 1
// int ExpToFieldElt[MultField + 1];   // A table that goes from the exponent of an element, in
//                                     // terms of a previously chosen generator of the
//                                     // multiplicative group, to its representation as a
//                                     // vector over GF[2]
// int FieldEltToExp[MultField + 1];   // The table that goes from the vector representation of
//                                     // an element to its exponent of the generator.
// int Bit[L];                         // An array of integers that is used to select individual bits:
//                                     // (A & Bit[i]) is equal to 1 if the i-th bit of A is 1, and 0
//                                     // otherwise.

// void rs_init_tables() {
//   for (int i = 0; i < L; i++) {
//     Bit[i] = 1 << i;
//   }

//   int alpha = 2;
//   int current = 1;

//   for (int i = 0; i < MultField; i++) {
//     ExpToFieldElt[i] = current;
//     FieldEltToExp[current] = i;
//     current *= alpha;
//     if (current >= (1 << L)) {
//       current ^= 0x11D;
//     }
//   }
// }

// Encoder rs_create_encoder(int Mpackets, int Rpackets, size_t packet_size) {
//   Encoder encoder;
//   encoder.Mpackets = Mpackets;
//   encoder.Rpackets = Rpackets;
//   encoder.Nsegs = packet_size / L;
//   return encoder;
// }


// Decoder rs_create_decoder(int Mpackets, int Rpackets, size_t packet_size, int num_orig_packets_received) {
//   Decoder decoder;
//   decoder.Mpackets = Mpackets;
//   decoder.Rpackets = Rpackets;
//   decoder.Nsegs = packet_size / L;
//   decoder.Nextra = Mpackets - num_orig_packets_received;
//   return decoder;
// }


// void rs_encode(Encoder &enc, uint8_t *orig_data, uint8_t *redundant_data) {
//   /// The number of rows in our Cauchy matrix is equal to the number of redundant packets
//   for (int row = 0; row < enc.Rpackets; row++) {
//     uint8_t *packet = redundant_data + row * enc.Nsegs * L;

//     /// The number of columns in our Cauchy matrix is equal to the number of informaiton packets
//     for (int col = 0; col < enc.Mpackets; col++) {
//       /// exponent is the multiplicative exponent of the element of the Cauchy matrix
//       int exponent = (MultField - FieldEltToExp[row ^ col ^ MultField]) % MultField;
      
//       /// Each element of our finite field is now represented as a Lfield by Lfield 0-1 matrix.
//       for (int row_bit = 0; row_bit < L; row_bit++) {
//         uint8_t *local_packet = packet + row_bit * enc.Nsegs;
        
//         for (int col_bit = 0; col_bit < L; col_bit++) {
//           if (ExpToFieldElt[exponent + row_bit] & Bit[col_bit]) {
//             uint8_t *local_orig_data = orig_data + col * enc.Nsegs * L + col_bit * enc.Nsegs;

//             for (int segment = 0; segment < enc.Nsegs; segment++) {
//               local_packet[segment] ^= local_orig_data[segment];
//             }
//           }
//         }
//       }
//     }
//   }
// }



// void rs_decode(Decoder &dec, const uint8_t *inp_recv_orig_data, const uint8_t *inp_recv_redund_data, uint8_t *outp_data, uint32_t *recv_idx, uint32_t *inv_mat) {
//   // Step 1: Collect RowInd and ColInd
//   std::vector<int> RowInd;
//   std::vector<int> ColInd;
//   int i = 0;
//   for (; i < dec.Mpackets; i++) {
//     if (recv_idx[i]) {
//       ColInd.push_back(i);
//     }
//   }

//   for (; i < dec.Mpackets + dec.Rpackets; i++) {
//     if (recv_idx[i]) {
//       RowInd.push_back(i - dec.Mpackets);
//     }
//   }

//   // Step 2: Invert the Cauchy matrix
//   rs_invert_cauchy_matrix(inv_mat, dec.Nextra, RowInd, ColInd);

//   ///////////// TO FIX
//   uint8_t *M = static_cast<uint8_t*>(malloc(dec.Nextra * dec.Nsegs * L));
//   memset(M, 0, dec.Nextra * dec.Nsegs * L);


//   // Step 3: Decode the message using the inverted matrix
//   // The data that was received is already stored in outp_data beforehand
//   for (int row = 0; row < dec.Nextra; row++) {
//     for (int col = 0; col < dec.Mpackets; col++) {
//       // if the message packet was received, then process it
//       if (recv_idx[col]) {
//         int exponent = (MultField - FieldEltToExp[RowInd[row] ^ col ^ MultField]) % MultField;
//         for (int row_bit = 0; row_bit < L; row_bit++) {
//           for (int col_bit = 0; col_bit < L; col_bit++) {
//             if (ExpToFieldElt[exponent + row_bit] & Bit[col_bit]) {
//               for (int segment = 0; segment < dec.Nsegs; segment++) {
//                 // Todo: check M access
//                 M[(row_bit + row*L)*dec.Nsegs + segment] ^= inp_recv_orig_data[(col_bit+col*L) * dec.Nsegs + segment];
//               }
//             }
//           }
//         }

//       }
//     }
//   }

//   // Step 4: Multiply the inverted matrix with the received data
//   for (int row = 0; row < dec.Nextra; row++) {
//     for (int col = 0; col < dec.Nextra; col++) {
//       uint32_t exponent = inv_mat[row * dec.Nextra + col];

//       for (int row_bit = 0; row_bit < L; row_bit++) {
//         for (int col_bit = 0; col_bit < L; col_bit++) {
//           if (ExpToFieldElt[exponent + row_bit] & Bit[col_bit]) {
//             for (int segment = 0; segment < dec.Nsegs; segment++) {
//               outp_data[row_bit * dec.Nsegs + ColInd[row] * L * dec.Nsegs + segment] ^= M[(col_bit + col * L)*dec.Nsegs + segment];
//             }
//           }
//         }
//       }
//     }
//   }
// }

// void rs_invert_cauchy_matrix(uint32_t *inv_mat, int Nextra, std::vector<int> &RowInd, std::vector<int> &ColInd) {
//   std::vector<int> C(Nextra, 0), D(Nextra, 0), E(Nextra, 0), F(Nextra, 0);
//   for (int row = 0; row < Nextra; row++) {
//     for (int col = 0; col < Nextra; col++) {

//       if (col != row) {
//         C[row] += FieldEltToExp[RowInd[row] ^ RowInd[col]];
//         D[col] += FieldEltToExp[ColInd[row] ^ ColInd[col]];
//       }

//       E[row] += FieldEltToExp[RowInd[row] ^ ColInd[col] ^ MultField];
//       F[col] += FieldEltToExp[RowInd[row] ^ ColInd[col] ^ MultField];
//     }
//   }

//   for (int row = 0; row < Nextra; row++) {
//     for (int col = 0; col < Nextra; col++) {
//       inv_mat[row * Nextra + col] = E[col] + F[row] - C[col] - D[row] - FieldEltToExp[RowInd[col] ^ ColInd[row] ^ MultField];

//       if (inv_mat[row*Nextra + col] >= 0) {
//         inv_mat[row*Nextra + col] %= MultField;
//       } else {
//         inv_mat[row*Nextra + col] = (MultField - ((-inv_mat[row*Nextra + col]) % MultField)) % MultField;
//       }
//     }
//   }
// }
