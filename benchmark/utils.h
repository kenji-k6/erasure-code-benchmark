#ifndef UTILS_H
#define UTILS_H
#include <cstddef> // for size_t

struct ecc_config {
  size_t payloadSize;
  unsigned int parityBits;
  bool allowSimd;
  bool allowVectorization;
};
#endif // UTILS_H