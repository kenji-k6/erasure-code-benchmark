#ifndef XOREC_AVX2_BM_HPP
#define XOREC_AVX2_BM_HPP

#include "abstract_bm.hpp"

class XorecBenchmarkAVX2 : public ECBenchmark {
public:
explicit XorecBenchmarkAVX2(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmarkAVX2() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
protected:
  uint8_t* m_parity_buffer;
};

#endif // XOREC_AVX2_BM_HPP