#ifndef WIREHAIR_BENCHMARK_H
#define WIREHAIR_BENCHMARK_H

#include "abstract_benchmark.h"
#include "wirehair/wirehair.h"


/**
 * @class WirehairBenchmark
 * @brief Benchmark implementation for the Wirehair ECC library https://github.com/catid/wirehair
 * 
 * This class implements the ECCBenchmark interface, providing specific functionality
 * for benchmarking the Wirehair library. It supports setup, teardown, encoding, decoding,
 * data loss simulation, corruption checking and cache flushing.
 * 
 * @attention Since Wirehair uses a fountain code approach, this means that recovery blocks are generated
 * until the data can be restored. To still have an adequate comparison with the other libraries, the
 * benchmark will limit the number of recovery blocks that Wirehair is allowed to generate.
 * For the same reason, part of the loss simulation has to be done in the decode function unfortunately.
 */
class WirehairBenchmark : public ECCBenchmark {
public:
  explicit WirehairBenchmark(const BenchmarkConfig& config) noexcept;
  ~WirehairBenchmark() noexcept = default;

  int setup() noexcept override;
  void teardown() noexcept override;
  int encode() noexcept override;
  int decode() noexcept override;
  void simulate_data_loss() noexcept override;
  bool check_for_corruption() const noexcept override;

private:
  uint8_t *original_buffer_ = nullptr;    ///< Buffer for the original data we want to transmit
  uint8_t *encode_buffer_ = nullptr;      ///< Buffer for the encoded data
  uint8_t *decode_buffer_ = nullptr;      ///< Buffer for the decoded data

  WirehairCodec encoder_ = nullptr;       ///< Wirehair encoder instance
  WirehairCodec decoder_ = nullptr;       ///< Wirehair decoder instance
};

#endif // WIREHAIR_BENCHMARK_H
