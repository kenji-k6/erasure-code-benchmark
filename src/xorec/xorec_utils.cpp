#include "xorec_utils.hpp"
#include "utils.hpp"

std::string get_version_name(XorecVersion version) {
  switch (version) {
    case XorecVersion::Scalar:
      return "Scalar";
    case XorecVersion::AVX:
      return "AVX";
    case XorecVersion::AVX2:
      return "AVX2";
  }
  throw_error("Invalid XorecVersion");
}

std::array<uint8_t, XOREC_MAX_DATA_BLOCKS> COMPLETE_DATA_BITMAP = {0};