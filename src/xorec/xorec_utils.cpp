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
    case XorecVersion::AVX512:
      return "AVX512";
  }
  throw_error("Invalid XorecVersion");
}