#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <cstddef>

/*
 * BenchmarkConfig: Configuration parameters for the benchmark
*/

struct BenchmarkConfig {
  // Common parameters
  size_t data_size;             // Total size of original data
  size_t block_size;            // Size of each block
  double redudancy_ratio;       // Recovery blocks / original blocks ratio
  double loss_rate;             // Simulated data loss rate
  int iterations;               // Number of iterations to run the benchmark

  struct {                      // Derived value (calculated during setup)
    size_t original_blocks;
    size_t recovery_blocks;
  } computed;
}; // struct BenchmarkConfig


/*
 * ECCBenchmark: Interface that all ECC libraries will implement
*/
class ECCBenchmark {
public:
  virtual ~ECCBenchmark() = default;

  // Initialize the benchmark with the given configuration
  virtual int setup(const BenchmarkConfig& config) = 0;

  // Run the encoding process
  virtual int encode() = 0;

  // Run the decoding process (with simulated data loss)
  virtual int decode(double loss_rate) = 0;

  // Cleanup the benchmark
  virtual void teardown() = 0;

  // Metrics collected during the benchmark
  struct Metrics {
    long long encode_time_us;
    long long decode_time_us;
    double encode_input_throughput_mbps;
    double encode_output_throughput_mbps;
    double decode_input_throughput_mbps;
    double decode_output_throughput_mbps;
  };

  // Get the metrics collected during the benchmark
  virtual Metrics get_metrics() const = 0;
}; // class ECCBenchmark

#endif // BENCHMARK_H