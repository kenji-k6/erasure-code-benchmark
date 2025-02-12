#include "benchmark_runner.h"



class Benchmark {
  public:
    PCGRandom rng;

    Benchmark() : rng(0, 0) {}

    void generate_random_data(void* buffer, size_t size) {
      uint8_t* byte_buffer = static_cast<uint8_t*>(buffer);
      for (size_t i = 0; i < size; i++) {
        byte_buffer[i] = rng.next() & 0xFF;
      }
    }


    void benchmark_leopard_rs(size_t buffer_size, unsigned original_count, unsigned recovery_count) {
      if (leo_init()) {
        exit(1);
      }

      unsigned work_count = leo_encode_work_count(original_count, recovery_count);
      assert(work_count > 0 && "Invalid input to leo_encode_work_count");

      // Create buffers
      std::vector<void*> original_data(original_count);
      std::vector<void*> recovery_data(recovery_count);
      std::vector<void*> work_data(work_count);

      // Allocate memory for buffers
      for (unsigned i = 0; i < original_count; i++) {
        original_data[i] = malloc(buffer_size);
        generate_random_data(original_data[i], buffer_size);
      }

      for (unsigned i = 0; i < recovery_count; i++) {
        recovery_data[i] = malloc(buffer_size);
      }

      for (unsigned i = 0; i < work_count; i++) {
        work_data[i] = malloc(buffer_size);
      }

      long long start_time = get_current_time_us();

      //Encode
      LeopardResult encode_result = leo_encode(buffer_size, original_count, recovery_count, work_count, original_data.data(), work_data.data());

      long long end_time = get_current_time_us();

      if (encode_result == Leopard_Success) {
        std::cout << "Encoding successful. Time: " << (end_time - start_time) << " microseconds\n";
      } else {
        std::cerr << "Encoding failed. Error: " << leo_result_string(encode_result) << std::endl;
      }


      // Decoding
      unsigned work_count2 = leo_decode_work_count(original_count, recovery_count);
      std::vector<void*> work_data_decode(work_count2);
      
      for (unsigned i = 0; i < work_count2; i++) {
        work_data_decode[i] = malloc(buffer_size);
      }

      start_time = get_current_time_us();
      LeopardResult decode_result = leo_decode(buffer_size, original_count, recovery_count, work_count2, original_data.data(), work_data.data(), work_data_decode.data()); 
      end_time = get_current_time_us();

      if (decode_result == Leopard_Success) {
        std::cout << "Decoding successful. Time: " << (end_time - start_time) << " microseconds\n";
      } else {
        std::cerr << "Decoding failed. Error: " << leo_result_string(decode_result) << std::endl;
      }
      
      std::cout << "No Seg" << '\n';
      // Free memory
      for (unsigned i = 0; i < original_count; i++) {
        free(original_data[i]);
      }

      std::cout << "No Seg" << '\n';
      for (unsigned i = 0; i < recovery_count; i++) {
        free(recovery_data[i]);
      }
      std::cout << "No Seg" << '\n';
      for (unsigned i = 0; i < work_count; i++) {
        free(work_data[i]);
      }
      std::cout << "No Seg" << '\n';
      for (unsigned i = 0; i < work_count2; i++) {
        free(work_data_decode[i]);
      }
    }
};




int main() {
  size_t buffer_size = 64000; // 1KB buffer
  unsigned original_count = 100; // 10 original data buffers
  unsigned recovery_count = 20; // 5 recovery data buffers

  Benchmark benchmark;
  benchmark.benchmark_leopard_rs(buffer_size, original_count, recovery_count);

  return 0;
}