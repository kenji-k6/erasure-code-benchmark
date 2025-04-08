#ifndef XOREC_AVX_BM_HPP
#define XOREC_AVX_BM_HPP

#include "abstract_bm.hpp"

class XorecBenchmarkAVX : public ECBenchmark {
public:
explicit XorecBenchmarkAVX(const BenchmarkConfig& config) noexcept;
  ~XorecBenchmarkAVX() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
protected:
  uint8_t* m_parity_buffer;
};

#endif // XOREC_AVX_BM_HPP