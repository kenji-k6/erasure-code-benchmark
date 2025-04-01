#ifndef WIREHAIR_BM_HPP
#define WIREHAIR_BM_HPP

#include "abstract_bm.hpp"
#include "wirehair/wirehair.h"


/**
 * @class WirehairBenchmark
 * @brief Benchmark implementation for the Wirehair EC library https://github.com/catid/wirehair
 * 
 * This class implements the ECBenchmark interface, providing specific functionality
 * for benchmarking the Wirehair library. It supports setup, teardown, encoding, decoding,
 * data loss simulation and corruption checking.
 * 
 * @attention Since Wirehair uses a fountain code approach, this means that recovery blocks are generated
 * until the data can be restored. To still have an adequate comparison with the other libraries, the
 * benchmark will limit the number of recovery blocks that Wirehair is allowed to generate.
 * For the same reason, part of the loss simulation has to be done in the decode function unfortunately.
 */
class WirehairBenchmark : public ECBenchmark {
public:
  explicit WirehairBenchmark(const BenchmarkConfig& config) noexcept;
  ~WirehairBenchmark() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;

private:
  uint8_t* m_encode_buf;
  WirehairCodec m_encoder = nullptr;       ///< Wirehair encoder instance
  WirehairCodec m_decoder = nullptr;       ///< Wirehair decoder instance
};

#endif // WIREHAIR_BM_HPP
