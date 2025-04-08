#ifndef XOREC_AVX512_BM_HPP
#define XOREC_AVX512_BM_HPP

#include "abstract_bm.hpp"

class XorecBenchmarkAVX512 : public ECBenchmark {
public:
explicit XorecBenchmarkAVX512(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmarkAVX512() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
protected:
  uint8_t* m_parity_buffer;
};

#endif // XOREC_AVX512_BM_HPP