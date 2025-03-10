#ifndef WIREHAIR_BENCHMARK_H
#define WIREHAIR_BENCHMARK_H

#include "abstract_benchmark.h"
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
  bool check_for_corruption() const noexcept override;

private:
  uint32_t num_total_blocks_;

  std::unique_ptr<uint8_t[]> original_buffer_;    ///< Buffer for the original data we want to transmit
  std::unique_ptr<uint8_t[]> encode_buffer_;      ///< Buffer for the encoded data
  std::unique_ptr<uint8_t[]> decode_buffer_;      ///< Buffer for the decoded data

  WirehairCodec encoder_;       ///< Wirehair encoder instance
  WirehairCodec decoder_;       ///< Wirehair decoder instance
};

#endif // WIREHAIR_BENCHMARK_H
